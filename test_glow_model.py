"""Unit tests for glow model (out of date)"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import glow_model


class TestLayers(tf.test.TestCase):
    """tests layers in glow model"""
    # TODO: add tests that determinant scale isn't influenced by batch size
    def _helper_train_layer(self, layer, image_shape=None, runs_with_null_grads=None,
                            learning_rate=1e-7):
        """sanity checks training of the layer"""
        if image_shape is None:
            image_shape = [4, 64, 64, 3]
        np.random.seed(1)
        input_image = tf.to_float(np.random.uniform(size=image_shape))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        means = tf.reshape(tf.to_float(np.random.uniform(size=image_shape)), [image_shape[0], -1])
        target_dist = tfp.distributions.MultivariateNormalDiag(loc=means,
                                                               scale_diag=np.ones_like(means))
        for i in range(0, 5):
            with tf.GradientTape() as tape:
                layer_output = tf.reshape(layer(input_image, determinant=False),
                                          [image_shape[0], -1])
                objective = -tf.reduce_sum(target_dist.log_prob(layer_output))
                if i > 0:
                    self.assertLess(objective, old_objective)
                old_objective = objective
            grads = tape.gradient(objective, layer.trainable_variables)
            if not runs_with_null_grads or i not in runs_with_null_grads:
                for grad in grads:
                    self.assertGreater(tf.norm(grad), 0)
            optimizer.apply_gradients(zip(grads, layer.trainable_variables))

    def test_InvertibleConv2D(self):
        """tests InvertibleConv2D layer"""
        np.random.seed(1)
        image_shape = [4, 64, 64, 3]
        input_image = tf.to_float(np.random.uniform(size=image_shape))

        layer_invertibleconv2d = glow_model.InvertibleConv2D(use_bias=False)

        layer_output, log_det = layer_invertibleconv2d(input_image,
                                                       determinant=True,
                                                       inverse=False)

        self.assertEqual(len(layer_invertibleconv2d.variables), 1)
        self.assertEqual(layer_invertibleconv2d.variables[0].shape.as_list(),
                         [image_shape[-1], image_shape[-1]])
        self.assertEqual(layer_output.shape.as_list(), image_shape)
        self.assertEqual(log_det.shape.as_list(), [])

        reconstructed_image = layer_invertibleconv2d(layer_output, inverse=True)

        with self.session():
            tf.get_default_session().run(tf.initializers.global_variables())
            self.assertAllClose(log_det.eval(), 0.0)
            self.assertAllClose(input_image.eval(), reconstructed_image.eval())
            self.assertAllClose(tf.norm(layer_output).eval(), tf.norm(input_image).eval())

    def test_InvertibleConv2D_train(self):
        """tests InvertibleConv2D layer during training"""
        layer_invertibleconv2d = glow_model.InvertibleConv2D(use_bias=False)
        self._helper_train_layer(layer_invertibleconv2d)

    def test_ActNorm(self):
        """tests InvertibleConv2D layer"""
        np.random.seed(1)
        image_shape = [4, 64, 64, 3]
        input_image = tf.to_float(np.random.uniform(size=image_shape))

        layer_actnorm = glow_model.ActNorm()

        layer_output, log_det = layer_actnorm(input_image,
                                              determinant=True,
                                              inverse=False,
                                              logscale_factor=3.)

        self.assertEqual(len(layer_actnorm.variables), 3)
        self.assertEqual(layer_actnorm.variables[0].shape.as_list(), [image_shape[3]])
        self.assertEqual(layer_actnorm.variables[1].shape.as_list(), [image_shape[3]])
        self.assertEqual(layer_output.shape.as_list(), image_shape)
        self.assertEqual(log_det.shape.as_list(), [])

        # TODO: correct tests (normalize by uncentered variance, then offset)
        batch_mean, batch_var = tf.nn.moments(input_image, axes=[0, 1, 2])
        self.assertAllClose(layer_actnorm.log_scale.numpy()*3.,
                            tf.log(tf.rsqrt(batch_var+layer_actnorm.epsilon)).numpy())
        self.assertAllClose(layer_actnorm.offset.numpy(), batch_mean.numpy())

        output_mean, output_var = tf.nn.moments(layer_output, axes=[0, 1, 2])
        self.assertAllClose(output_mean, [0.]*image_shape[-1], rtol=1e-4, atol=1e-4)
        self.assertAllClose(output_var, [1.]*image_shape[-1], rtol=1e-4, atol=1e-4)

        reconstructed_image = layer_actnorm(layer_output, inverse=True)
        self.assertAllClose(input_image, reconstructed_image)
        self.assertEqual(layer_actnorm.first_call.numpy(), False)
        input_image_next_batch = tf.to_float(np.random.uniform(size=image_shape,
                                                               low=10, high=100))
        layer_output_next_batch = layer_actnorm(input_image_next_batch,
                                                determinant=False,
                                                inverse=False)
        output_mean, output_var = tf.nn.moments(layer_output_next_batch, axes=[0, 1, 2])
        self.assertGreater(output_mean[0], 100.)
        self.assertGreater(output_var[0], 1000.)

    def test_ActNorm_train(self):
        """tests ActNorm layer during training"""
        layer_actnorm = glow_model.ActNorm()
        self._helper_train_layer(layer_actnorm)

    # TODO: complete
    def test_CouplingLayer(self):
        np.random.seed(1)
        image_shape = [4, 64, 64, 3]
        input_image = tf.to_float(np.random.uniform(size=image_shape))

        layer_coupling = glow_model.CouplingLayer(additive=False)

        layer_output, log_det = layer_coupling(input_image,
                                               determinant=True,
                                               inverse=False)
        #self.assertEqual(len(layer_coupling.variables), 4)

    def test_CouplingLayer_train(self):
        """tests ActNorm layer during training"""
        layer_coupling = glow_model.CouplingLayer(additive=False)
        self._helper_train_layer(layer_coupling, image_shape=[4, 64, 64, 8],
                                 runs_with_null_grads=[0])
        layer_coupling = glow_model.CouplingLayer(additive=True)
        self._helper_train_layer(layer_coupling, image_shape=[4, 64, 64, 8],
                                 runs_with_null_grads=[0])

    def test_GlowBlock_train(self):
        """tests GlowBlock layer during training"""
        layer_coupling = glow_model.GlowBlock(additive=False)
        self._helper_train_layer(layer_coupling, image_shape=[4, 64, 64, 8],
                                 runs_with_null_grads=[0])
        layer_coupling = glow_model.GlowBlock(additive=True)
        self._helper_train_layer(layer_coupling, image_shape=[4, 64, 64, 8],
                                 runs_with_null_grads=[0])

    def test_GlowModel_train(self):
        """tests GlowModel layer during training"""
        model = glow_model.GlowModel(2, 2, additive=False)
        self._helper_train_layer(model, image_shape=[4, 64, 64, 8],
                                 runs_with_null_grads=[0])
        model = glow_model.GlowModel(2, 2, additive=True)
        self._helper_train_layer(model, image_shape=[4, 64, 64, 8],
                                 runs_with_null_grads=[0])

    # TODO: test inverse works properly
    def test_GlowModel(self):
        np.random.seed(1)
        image_shape = [4, 64, 64, 3]
        input_image = tf.to_float(np.random.uniform(size=image_shape))
        model = glow_model.GlowModel(2, 2, additive=False)
        reconstructed = model.sample(model(input_image, determinant=False))
        self.assertAllClose(input_image, reconstructed)

        self._helper_train_layer(model, image_shape=image_shape,
                                 runs_with_null_grads=[0])
        reconstructed = model.sample(model(input_image, determinant=False))
        self.assertAllClose(input_image, reconstructed)


if __name__ == "__main__":
    tf.enable_eager_execution()
    tf.test.main()
