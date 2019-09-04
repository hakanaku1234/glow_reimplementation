import tensorflow as tf
import tensorflow_probability as tfp
import argparse
import numpy as np
import sys, shutil, time
import datetime
import os
import contextlib
from glow_model import *
from ops import *

def get_init_actnorm_op(model, iterator, num_iterations):
    if num_iterations > 0:
        samples = []
        with tf.name_scope("actnorm_init_batch"):
            for i in range(0, num_iterations):
                samples.append(iterator.get_next())
            initial_batch = tf.concat(samples, axis=0)
        init_actnorm_op = model(initial_batch) # models should(?) be able to take very large batch sizes if no need to track intermediates
        return init_actnorm_op
    else:
        return tf.no_op()
    
def train(args):
    print(args)
    image_h, image_w = [args.image_dim, args.image_dim]
    minibatch_size = args.minibatch_size
    initial_minibatch_size = args.initial_minibatch_size
    decay_learning_rate = args.decay_learning_rate
    eager = args.eager
    resume = args.resume
    log_dir = args.log_dir
    sample_dir = args.sample_dir
    args.warmup_batches = max(args.warmup_batches, 1)
    if args.num_minibatches is None:
        num_minibatches = sys.maxsize**10
    else:
        num_minibatches = args.num_minibatches
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_device
    if eager:
        tf.enable_eager_execution()
    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)

    if tf.executing_eagerly():
        it = get_dataset(image_h, image_w, sample_dir, args.n_bits, minibatch_size=minibatch_size)
        sess = None
    else:
        sess = tf.Session()
        with tf.name_scope("dataset_processing_%d_%d"%(image_h, image_w)):
            it, init_op_it = get_dataset(image_h, image_w, sample_dir, args.n_bits, minibatch_size=minibatch_size)
        
    model = GlowModel(L=args.L, K=args.K, additive=args.coupling is 'additive', coupling_filters=args.channels)
    
    with tf.contrib.eager.restore_variables_on_create(tf.train.latest_checkpoint(log_dir) if resume else None) if tf.executing_eagerly() else contextlib.suppress():
        summary_writer = tf.contrib.summary.create_file_writer(args.log_dir, flush_millis=10000)

        step = tf.train.get_or_create_global_step()
        tensors_to_log = None
        
        init_actnorm_op = get_init_actnorm_op(model, it, args.initial_minibatch_size // args.minibatch_size)

        if tf.executing_eagerly():
            optim = tf.train.AdamOptimizer(lambda: args.learning_rate * min(1., step / args.warmup_batches))
            # test reconstructing a sample:
            #reconstructed = model.sample(model(it.get_next())[0])
            #savearray(np.uint8(127.5*(reconstructed.numpy()[0]+1)), os.path.join(log_dir, "sample_test.png"))
        else:
            with summary_writer.as_default() if args.summary else contextlib.suppress(), tf.contrib.summary.always_record_summaries() if args.summary else contextlib.suppress():
                x = it.get_next()
                [z, determinants] = model(x)
                
                #from tensorflow.python import debug as tf_debug
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                with tf.name_scope("prior"):
                    prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(z), scale_diag=tf.ones_like(z), name="MVD")# prior on latents p_z (not prior on parameters p_theta used for MAP)
                
                #objective_tensor = (tf.reduce_mean(0.5*(tf.reduce_sum(tf.square(z), axis=-1))) + -tf.reduce_sum(determinants))/(image_h*image_w) # check this
                with tf.name_scope("objective"):
                    objective_tensor = (-(tf.reduce_mean(prior.log_prob(z) + determinants)) / (image_h*image_w))
                with tf.name_scope("reconstructed"):
                    temperature = tf.placeholder(dtype=tf.float32, shape=())
                    sample_op = model.sample(tf.random_normal([minibatch_size, image_h*image_w*3], 0., temperature))
                    reconstruct_op = model.sample(z)
                    sample_grid = tf.contrib.gan.eval.image_grid(
                                                                 tf.reshape(sample_op, [minibatch_size, image_h, image_w, 3]),
                                                                 [4 if minibatch_size > 4 else minibatch_size, max(minibatch_size//4, 1)],
                                                                 image_shape=(image_h, image_w),
                                                                 num_channels=3
                                                                )
                
                if args.summary:
                    tf.contrib.summary.initialize(graph=tf.get_default_graph(), session=sess)
                with tf.name_scope("optimization"):
                    learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=())
                    # optim = tf.train.AdamOptimizer(learning_rate_ph)
                    # optim = tf.train.GradientDescentOptimizer(learning_rate_ph)
                    # grads = optim.compute_gradients(objective_tensor, var_list = model.variables)
                    # if args.summary:
                    #     for grad, var in grads:
                    #         tf.contrib.summary.histogram(name=var.name, tensor=grad)
                    #         tf.contrib.summary.scalar(name=var.name+"_variance",
                    #                                   tensor=tf.nn.moments(grad, axes=[0, 1, 2, 3])[1])
                    # train_step = optim.apply_gradients(grads)
                    #optim = tf.keras.optimizers.Adamax(lr=learning_rate_ph)
                    #train_step = optim.get_updates(loss=objective_tensor, params=model.trainable_variables)
                    train_step, _, _ = ops.adamax(params=model.trainable_variables,
                                                  cost_or_grads=objective_tensor,
                                                  alpha=learning_rate_ph)
                    # train_step = optim.minimize(objective_tensor, var_list = model.variables)
                update_step_op = step.assign_add(1)
                if args.log_tensor_regex is not None:
                    tensors_to_log = log_ops(args.log_tensor_regex.split(" "))
                summary_ops = tf.contrib.summary.all_summary_ops()
                sess.run((tf.local_variables_initializer(), tf.global_variables_initializer()))
                sess.run(init_op_it)
                sess.run([init_actnorm_op, summary_ops])
                # var_list_save = model.variables + model.block_variables + optim.variables() + [step]
                var_list_save = model.variables + [step] #+ optim.variables()
                # needs to be run after AdamOptimizer is added to graph to save adam parameters
                saver = tf.train.Saver(max_to_keep=2, var_list=var_list_save)
                
                if resume:
                    load_result, batch_counter = load(log_dir, saver, sess=sess)
                    if not load_result:
                        print("could not load")
                        sys.exit(0)
                        
            # tf.get_default_graph().finalize()
        start = time.time()
        # n batch init generally set to larger to get an accurate initialization
        with summary_writer.as_default() if (args.eager and args.summary) else contextlib.suppress(), tf.contrib.summary.always_record_summaries() if (args.eager and args.summary) else contextlib.suppress():
            for i in range(0, num_minibatches):
                if tf.executing_eagerly():
                    x = it.get_next()
                    with tf.GradientTape() as tape:
                        [z, determinants] = model(x)
                        # prior on latents p_z (not prior on parameters p_theta used for MAP)
                        prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(z),
                                                                         scale_diag=tf.ones_like(z))
                        # print(tf.reduce_sum(determinants))
                        objective = -(tf.reduce_mean(prior.log_prob(z) + determinants) / (image_h*image_w))
                        # objective = (tf.reduce_mean(0.5*(tf.reduce_sum(tf.square(z), axis=-1))) +
                        # -tf.reduce_sum(determinants))/(image_h*image_w) # check this
                    grads = tape.gradient(objective, model.variables)
                    optim.apply_gradients(zip(grads, model.variables))
                    step_integer = step.numpy()
                    step.assign_add(1)
                else:
                    step_integer = sess.run(step)
                    ops_to_run = {"train_step": train_step, "objective_tensor": objective_tensor, "z": z}
                    if args.summary:
                        ops_to_run["summary"] = summary_ops
                    if tensors_to_log is not None:
                        ops_to_run["tensors_to_log"] = tensors_to_log
                    #grads = tf.gradients(objective_tensor, model.variables).eval(sess)
                    #import pdb; pdb.set_trace()
                    run_result = sess.run(ops_to_run,
                                          feed_dict={learning_rate_ph: args.learning_rate *
                                                     min(1., step_integer/args.warmup_batches)})
                    # _, objective, z_values = sess.run([train_step, objective_tensor, z],
                    # feed_dict = {learning_rate_ph:args.learning_rate*min(1., step_integer/args.warmup_batches)})
                    objective = run_result["objective_tensor"]
                    z_values = run_result["z"]
                    # determinants_result = run_result["determinants"]
                    
                    if tensors_to_log is not None:
                        for index,tensor in enumerate(tensors_to_log):
                            print(tensor.name + " : " + str(run_result["tensors_to_log"][index]))
                    # print(run_result["tensors_to_log"])
                    # print("det " + str(determinants_result))
                    # print(z_values)
                    # print("determinants")
                    # print(determinant_values)
                    sess.run(update_step_op)
                if i == 0:
                    print("total params: " + str(get_parameter_counts(model.variables)[0]))
                    pc = get_parameter_counts(tf.trainable_variables())[1]
                    accumulated = 0
                    counts = {}
                    for j, (count, name) in enumerate(pc):
                        print(str(count) + " " + name + " " + str(j))
                        accumulated += count
                        print("\t" + str(accumulated))
                        if count not in counts:
                            counts[count] = 0
                        counts[count] += 1
                    for key in sorted(counts, reverse=True):
                        print(str(key) + " : " + str(counts[key]))
                        
                    if tf.executing_eagerly():
                        var_list_save = model.variables + optim.variables()
                        # var_list_save = model.variables + model.block_variables + optim.variables()
                        # needs to be run after AdamOptimizer is added to graph to save adam parameters
                        saver = tf.train.Saver(max_to_keep=2, var_list=var_list_save)
                        
                    if tf.executing_eagerly():
                        reconstructed = model.sample(z)
                        savearray(np.uint8(tf.clip_by_value(
                            256.*tf.floor((reconstructed.numpy()[0]+.5)*(2.**args.n_bits))/(2.**args.n_bits), 0, 255)),
                            os.path.join(log_dir, "real_sample_%d.png" % step_integer))
                    else:
                        reconstructed = sess.run(reconstruct_op)
                        savearray(np.uint8(np.clip(
                            256.*np.floor((reconstructed[0]+.5)*(2.**args.n_bits))/(2.**args.n_bits), 0, 255)),
                            os.path.join(log_dir, "real_sample_%d.png" % step_integer))
                if i == 4:
                    if args.save_initial_state:
                        name_dict = shareable_variable_name_dict(tf.trainable_variables())
                        saver_init = tf.train.Saver(max_to_keep=1, var_list=name_dict)
                        saver_init.save(sess = sess, save_path=os.path.join(log_dir, "model_init"), global_step=0 )
                minutes = (time.time()-start)/60
                print("step: %d: minutes: %d objective %f"%(step_integer, minutes, objective))
                if step_integer % 100 == 0:
                    print("sampling")
                    if tf.executing_eagerly():
                        sampled = model.sample(tf.random_normal([1, image_h*image_w*3], 0., args.temperature))
                        savearray(np.uint8(tf.clip_by_value(
                            256.*tf.floor((sampled.numpy()[0]+.5)*(2.**args.n_bits))/(2.**args.n_bits), 0, 255)),
                            os.path.join(log_dir, "temp_%.1f_sample_%d.png" % (args.temperature, step_integer)))
                        sampled = model.sample(tf.random_normal([1, image_h*image_w*3], 0., 0.))
                        savearray(np.uint8(tf.clip_by_value(
                            256.*tf.floor((sampled.numpy()[0]+.5)*(2.**args.n_bits))/(2.**args.n_bits), 0, 255)),
                            os.path.join(log_dir, "temp_%.1f_sample_%d.png" % (0., step_integer)))
                    else:
                        sampled = sess.run(sample_grid, feed_dict={temperature:args.temperature})
                        savearray(np.uint8(np.clip(
                            256.*np.floor((sampled[0]+.5)*(2.**args.n_bits))/(2.**args.n_bits), 0, 255)),
                            os.path.join(log_dir, "temp_%.1f_sample_%d.png" % (args.temperature, step_integer)))
                        sampled = sess.run(sample_grid, feed_dict={temperature: 0.})
                        savearray(np.uint8(np.clip(
                            256.*np.floor((sampled[0]+.5)*(2.**args.n_bits))/(2.**args.n_bits), 0, 255)),
                            os.path.join(log_dir, "temp_%.1f_sample_%d.png" % (0., step_integer)))
                        
                    # x = model(x, determinant=False)
                    # reconstructed = model.sample(x)
                    # savearray(np.uint8(127.5*(reconstructed.numpy()[0]+1)), os.path.join(log_dir, "sample_%d.png"%i))
                if step_integer % 1000 == 0:
                    if not tf.executing_eagerly():
                        plot_zs(z_values, os.path.join(log_dir, "z_plot%d.png"%(step_integer/1000)))
                    saver.save(sess=sess if not tf.executing_eagerly() else None,
                               save_path=os.path.join(log_dir, "model"), global_step=step_integer)
                    
                    if tf.executing_eagerly():
                        reconstructed = model.sample(z)
                        savearray(np.uint8(tf.clip_by_value(
                            256.*tf.floor((reconstructed.numpy()[0]+.5)*(2.**args.n_bits))/(2.**args.n_bits), 0, 255)),
                            os.path.join(log_dir, "real_sample_%d.png" % step_integer))
                    else:
                        reconstructed = sess.run(reconstruct_op)
                        savearray(np.uint8(np.clip(
                            256.*np.floor((reconstructed[0]+.5)*(2.**args.n_bits))/(2.**args.n_bits), 0, 255)),
                            os.path.join(log_dir, "real_sample_%d.png" % step_integer))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameters for glow training")

    # save command line to log dir
    parser.add_argument("--eager", action='store_true', default=False, help="use eager mode")
    parser.add_argument("--summary", action='store_true', default=False, help="record tensor summaries")
    parser.add_argument("--log_tensor_regex", type=str, default=None, help="logs tensors with names like [regex]")
    parser.add_argument("--temperature", type=float, default=.8, help="sampling temperature")
    parser.add_argument("--save_initial_state", action='store_true', default=False,
                        help="save the initial values for variables with generic names")

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--minibatch_size", type=int, default=16, help="minibatch size")
    optimization_group.add_argument("--seed", type=int, default=None,
                                    help="random seed, unfortunately some tensorflow ops aren't \
                                     entirely desterministic, divergence may occur with same seed")
    optimization_group.add_argument("--initial_minibatch_size", type=int, default=256,
                                    help="minibatch size for initializing actnorm")
    optimization_group.add_argument("--num_minibatches", type=int, default=None, help="number of batches to process")
    optimization_group.add_argument("--learning_rate", type=float, default=.001, help="initial learning rate")
    optimization_group.add_argument("--decay_learning_rate", action='store_true',
                                    default=False, help="should decay learning rate [not implemented]")
    optimization_group.add_argument("--warmup_batches", type=float, default=3000,
                                    help="scale learning rate from 0 during these batches")
    optimization_group.add_argument("--cuda_visible_device", type=str, default=None, help="set CUDA_VISIBLE_DEVICES")

    log_dir_type_group = parser.add_mutually_exclusive_group(required=False)

    log_dir_type_group.add_argument("--sample", action='store_true', default=False,
                                    help="resume from latest checkpoint in the log dir [not implemented]")
    log_dir_type_group.add_argument("--resume", action='store_true', default=False,
                                    help="resume from latest checkpoint in the log dir")
    log_dir_type_group.add_argument("--title", type=str,
                                    help="generate a new log directory containing the specified title")

    dir_group = parser.add_argument_group("directories")
    dir_group.add_argument("--log_dir", type=str, required=True,
                           help="directory containing logging directories, or log directory to \
                            resume from if --resume is used")
    dir_group.add_argument("--sample_dir", type=str, required=True,
                           help="directory containing sample images")
    dir_group.add_argument("--class_dir", type=str, required=False,
                           help="directory containing sample classes [not implemented]")

    arch_group = parser.add_argument_group("architecture")
    arch_group.add_argument("--image_dim", type=int, default=64, help="height/[width] of image")
    arch_group.add_argument("--K", type=int, default=32, help="number of steps per scale")
    arch_group.add_argument("--L", type=int, default=3, help="number of scales")
    arch_group.add_argument("--channels", type=int, default=512, help="NN channels")
    arch_group.add_argument("--coupling", type=str, default="additive", help="[additive|affine] type of coupling")
    arch_group.add_argument("--n_bits", type=int, default=5, help="bits used to represent input [not implemented]")
    
    args = parser.parse_args(sys.argv[1:])
    
    if args.title:
        args.log_dir = os.path.join(args.log_dir, args.title+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, "code")):
        shutil.copytree(os.path.dirname(os.path.realpath(__file__)), os.path.join(args.log_dir, "code"))
    
    with open(os.path.join(args.log_dir, "run_%s.txt"%datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), "w") as f:
        f.write(" ".join(sys.argv))
        
    if args.sample:
        sample(args)
    else:
        train(args)

