
import math
import tensorflow as tf
import ops

import numpy as np


class InvertibleConv2D(tf.keras.layers.Layer):
    """Invertible convolution for mixing channels in flow models"""

    def __init__(self, kernel_size=1, stride=None, use_bias=False, padding='SAME'):
        """
        Initialize InvertibleConv2D

        [TODO: extended description]

        Parameters
        ----------
        kernel_size : int
            kernel size of convolution, default of 1 is used for rotation along channels
        stride : int list
            stride of convolution, default of [1, 1, 1, 1] used for rotation along channels
        use_bias : boolean
            true if a per-channel bias should be added after the convolution
        padding : string
            convolution's padding
        """
        super().__init__()
        if stride is None:
            self.stride = [1, 1, 1, 1]
        else:
            self.stride = stride
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        if len(self.stride) == 2:
            self.stride = [1, self.stride[0], self.stride[1], 1]
        self.padding = padding

    def build(self, input_shape):
        """
        Build InvertibleConv2D to according to the specified shape

        [TODO: extended description]

        Parameters
        ----------
        input_shape : TensorShape
            shape of the input
        """
        channels = input_shape.as_list()[-1]
        w_init = tf.constant_initializer(np.linalg.qr(np.random.randn(channels, channels))[0].astype("float32"))
        self.filter_w = self.add_weight(name="filter_w", shape=(channels, channels),
                                        initializer=w_init)#ops.orthogonal_init)
        self.filter_w = tf.reshape(self.filter_w, [self.kernel_size, self.kernel_size,
                                   channels, channels])
        if self.use_bias:
            self.biases = self.add_weight(name="biases",
                                          shape=(channels),
                                          initializer=tf.zeros_initializer())

    def call(self, inputs, determinant=False, inverse=False):
        """
        Apply InvertibleConv2D to the specified input

        [TODO: extended description]

        Parameters
        ----------
        inputs : tensor (shape [num_batches, height, width, channels])
            apply convolution operation to this input
        determinant : bool
            true if this function should return the convolution's determinant as a second
            return value
        inverse : bool
            true if the function's inverse should be used

        Returns
        -------
        tensor (if determinant=False)
            Result of applying the convolution to the input
        tensor, tensor (if determinant=True)
            Result of applying the convolution to the input, along with the determinant
        """
        if not inverse:
            net = tf.nn.conv2d(inputs, self.filter_w, self.stride,
                               padding=self.padding,
                               name="1x1conv2d")
        else:
            net = tf.nn.conv2d(inputs, tf.matrix_inverse(self.filter_w), self.stride,
                               padding=self.padding,
                               name="1x1conv2d")
        if self.use_bias:
            net = tf.nn.bias_add(net, self.biases)
        if determinant:
            shape = inputs.get_shape().as_list()
            total_positions = shape[1]*shape[2]

            log_det = total_positions * \
                      tf.cast(
                              tf.log(tf.abs(tf.squeeze(tf.matrix_determinant(tf.cast(self.filter_w,
                                                                                    "float64"))))),
                              "float32")
            tf.contrib.summary.scalar("det", log_det)
            return [net, log_det]
        return net


class ActNorm(tf.keras.layers.Layer):
    """
    Normalize activations with data dependent initialization
    This allows for smaller batch sizes than batch normalization
    """
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.num_channels = input_shape.as_list()[-1]
        self.offset = self.add_weight(name="offset", shape=[1, 1, 1, self.num_channels], dtype=tf.float32,
                                      initializer=tf.zeros_initializer())
        self.log_scale = self.add_weight(name="log_scale", shape=[1, 1, 1, self.num_channels], dtype=tf.float32,
                                     initializer=tf.ones_initializer())
        self.first_call = self.add_variable(name="first_call", shape=[], dtype=tf.bool,
                                            initializer=tf.constant_initializer(True),
                                            trainable=False)

    def call(self, inputs, determinant=False, inverse=False, logscale_factor=1.):
        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=True)
        assign_ops = tf.cond(self.first_call,
                             true_fn=lambda: [self.log_scale.assign(tf.log(tf.rsqrt(
                                                                    batch_var+self.epsilon))
                                                                    /logscale_factor),
                                              self.offset.assign(batch_mean),
                                              self.first_call.assign(False)],
                             false_fn=lambda: [self.log_scale.assign(self.log_scale),
                                               self.offset.assign(self.offset),
                                               self.first_call.assign(False)])

        if not inverse:
            with tf.control_dependencies(assign_ops):
                normalized = (inputs - self.offset)*tf.exp(self.log_scale*logscale_factor)
        else:
            normalized = inputs/tf.exp(self.log_scale*logscale_factor) + self.offset
        if determinant:
            shape = inputs.get_shape().as_list()
            total_positions = shape[1]*shape[2]
            log_det = total_positions*tf.reduce_sum(self.log_scale*logscale_factor)
            
            tf.contrib.summary.scalar("det", log_det)
            tf.contrib.summary.histogram("log_scale", self.log_scale)
            return [normalized, log_det]
        return normalized


class CouplingLayer(tf.keras.Model):
    def __init__(self, additive=False, filters=512):
        super().__init__()
        self.additive = additive
        self.filters = filters

    def build(self, input_shape):
        self.input_channels = input_shape.as_list()[-1]
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters,
                                            kernel_size=3,
                                            strides=1,
                                            padding='VALID',
                                            use_bias=False,
                                            kernel_initializer=tf.random_normal_initializer(0., .05),
                                            name="conv1")
        self.actnorm1 = ActNorm() # paper doesn't mention ActNorm in NN, but makes sense
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters,
                                            kernel_size=1,
                                            strides=1,
                                            padding='SAME',
                                            use_bias=False,
                                            kernel_initializer=tf.random_normal_initializer(0., .05),
                                            name="conv2")
        self.actnorm2 = ActNorm() # paper doesn't mention ActNorm in NN, but makes sense
        # compare zeros initialization with openai implementation
        self.conv_add_scale = tf.keras.layers.Conv2D(filters=self.input_channels,
                                                     kernel_size=3,
                                                     strides=1,
                                                     kernel_initializer=tf.zeros_initializer(),
                                                     padding='VALID',
                                                     name="conv_add_scale")

    def call(self, inputs, determinant=False, inverse=False):
        num_channels = inputs.get_shape().as_list()[-1]
        # TODO: combine inverse/non inverse? beginning is very similar for both
        # and it's a source of error if they differ
        if not inverse:
            x_a = inputs[:, :, :, :num_channels//2]
            x_b = inputs[:, :, :, num_channels//2:]
            result = ops.add_edge_padding(x_a, filter_size=[3,3])
            result = self.conv1(result)
            # TODO: verify, non-openai implementation uses 'actnorms' which seem
            # to just be initialized with random_uniform and never changed
            result = self.actnorm1(result)
            result = tf.nn.relu(result)
            result = self.conv2(result)
            result = self.actnorm2(result)
            result = tf.nn.relu(result)
            result = ops.add_edge_padding(result, filter_size=[3,3])
            result_add_scale = self.conv_add_scale(result)
            # offset = result_add_scale[:, :, :, :self.input_channels//2]
            offset = result_add_scale[:, :, :, 0::2]
            if not self.additive:
                # log_scale = result_add_scale[:, :, :, self.input_channels//2:]
                log_scale = result_add_scale[:, :, :, 1::2]
                log_scale = tf.clip_by_value(log_scale, -15.0, 15.0)  # from openai implementation TODO: test
                s = tf.exp(log_scale)
                y_b = tf.multiply(s, x_b + offset)
                log_det = tf.reduce_sum(tf.log(s), [1, 2, 3])
            else:
                y_b = x_b + offset
                log_det = 0.
            y_a = x_a
            # TODO: CHECK ORDER
            result = tf.concat([y_a, y_b], axis=-1)
            if determinant:
                tf.contrib.summary.scalar("det", log_det)
                return [result, log_det] # don't sum over batches
            return result
        else:
            y_a = inputs[:, :, :, :num_channels//2]
            y_b = inputs[:, :, :, num_channels//2:]
            result = ops.add_edge_padding(y_a, filter_size=[3,3])
            result = self.conv1(result)
            result = self.actnorm1(result)
            result = tf.nn.relu(result)
            result = self.conv2(result)
            result = self.actnorm2(result)
            result = tf.nn.relu(result)
            result = ops.add_edge_padding(result, filter_size=[3,3])
            result_add_scale = self.conv_add_scale(result)
            # offset = result_add_scale[:, :, :, :self.input_channels//2]
            offset = result_add_scale[:, :, :, 0::2]
            if not self.additive:
                # log_scale = result_add_scale[:, :, :, self.input_channels//2:]
                log_scale = result_add_scale[:, :, :, 1::2]
                x_b = tf.multiply(tf.exp(-log_scale), y_b) - offset
            else:
                x_b = y_b - offset
            x_a = y_a
            result = tf.concat([x_a, x_b], axis=-1)
            return result


class GlowBlock(tf.keras.Model):
    def __init__(self, additive=False, coupling_filters=512):
        super().__init__()
        self.additive = additive
        self.coupling_filters = coupling_filters

    def build(self, input_shape):
        self.actnorm = ActNorm()
        self.conv1x1 = InvertibleConv2D()
        self.coupling = CouplingLayer(self.additive, self.coupling_filters)

    def call(self, net, determinant=False, inverse=False):
        if not inverse:
            if determinant:
                log_det_total = 0
                #tf.contrib.summary.scalar("pre_actnorm_variance", tf.reduce_mean(tf.nn.moments(net, [0, 1, 2])[1]))
                [net, log_det] = self.actnorm(net, determinant=True)
                #tf.contrib.summary.scalar("post_actnorm_variance", tf.reduce_mean(tf.nn.moments(net, [0, 1, 2])[1]))
                log_det_total += log_det
                [net, log_det] = self.conv1x1(net, determinant=True)
                tf.contrib.summary.scalar("post_conv1x1_variance", tf.reduce_mean(tf.nn.moments(net, [0, 1, 2])[1]))
                print("conv1x1 " + str(log_det))
                log_det_total += log_det
                [net, log_det] = self.coupling(net, determinant=True)
                tf.contrib.summary.scalar("post_coupling_variance", tf.reduce_mean(tf.nn.moments(net, [0, 1, 2])[1]))
                log_det_total += log_det
                return [net, log_det_total]
            else:
                net = self.actnorm(net)
                net = self.conv1x1(net)
                net = self.coupling(net)
                return net
        else:
            net = self.coupling(net, inverse=True)
            net = self.conv1x1(net, inverse=True)
            net = self.actnorm(net, inverse=True)
            return net


class GlowModel(tf.keras.Model):
    def __init__(self, L, K, additive=False, coupling_filters=512):
        super().__init__()
        self.additive = additive
        self.block_variables = None
        self.L = L
        self.K = K
        self.coupling_filters = coupling_filters

    def build(self, input_shape):
        self.blocks = [[GlowBlock(additive=self.additive, 
                        coupling_filters=self.coupling_filters)
                        for k in range(0, self.K)]
                       for l in range(0, self.L)]

    def sample(self, z, flat=True):
        z_reshaped = []
        num_samples = -1 #z.get_shape().as_list()[0]
        x = tf.reshape(z[:, :self.z_shape_list[-1][0]*
                         self.z_shape_list[-1][1]*
                         self.z_shape_list[-1][2]],
                       [num_samples] + list(self.z_shape_list[-1]))
        z = z[:, self.z_shape_list[-1][0]*self.z_shape_list[-1][1]*self.z_shape_list[-1][2]:]
        for l in reversed(range(1, self.L)):
            for k in reversed(range(0, self.K)):
                x = self.blocks[l][k](x, inverse=True)
            x = ops.unsqueeze2d(x)
            x_concat = tf.reshape(z[:, :self.z_shape_list[l-1][0]*
                                    self.z_shape_list[l-1][1]*
                                    self.z_shape_list[l-1][2]],
                                  [num_samples] + list(self.z_shape_list[l-1]))
            z = z[:, self.z_shape_list[l-1][0]*
                  self.z_shape_list[l-1][1]*
                  self.z_shape_list[l-1][2]:]
            x = tf.concat([x_concat, x], axis=-1)
        l = 0
        for k in reversed(range(0, self.K)):
            x = self.blocks[l][k](x, inverse=True)
        x_reconstructed = ops.unsqueeze2d(x)
        return x_reconstructed

    def call(self, inputs, determinant=True):
        det_list = []
        z_list = []
        self.z_shape_list = []
        x = inputs
        determinant_accumulator = tf.constant(0.)
        for l in range(0, self.L-1):
            x = ops.squeeze2d(x)
            with tf.name_scope("L_%d"%l):
                for k in range(0, self.K):
                    with tf.name_scope("K_%d"%k):
                        [x, det] = self.blocks[l][k](x, determinant=True)
                        determinant_accumulator += det
                    det_list.append(det)
                with tf.name_scope("extract_latents"):
                    num_channels = x.get_shape().as_list()[-1]
                    self.z_shape_list.append(x[:, :, :, :num_channels//2].get_shape().as_list()[1:])
                    z_list.append(tf.layers.flatten(x[:, :, :, :num_channels//2]))
                    x = x[:, :, :, num_channels//2:]
        x = ops.squeeze2d(x)
        l = max(self.L-1, 0)
        with tf.name_scope("L_%d"%l):
            for k in range(0, self.K):
                with tf.name_scope("K_%d"%k):
                    [x, det] = self.blocks[l][k](x, determinant=True)
                    determinant_accumulator += det
                det_list.append(det)

        self.z_shape_list.append(x.get_shape().as_list()[1:])
        z_list.append(tf.layers.flatten(x))

        # combine z values
        z_list.reverse()
        z = tf.concat(z_list, axis=-1, name="combine_multiscale_latents")
        # combine
        determinants = tf.stack(det_list, name="determinants")
        if determinant:
            return [z, determinant_accumulator]
        return z
