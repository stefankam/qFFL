import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    def __init__(self, num_classes, q, optimizer, seed=1):

        # params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            (self.features,
             self.labels,
             self.is_training,
             self.train_op,
             self.grads,
             self.eval_metric_ops,
             self.loss,
             self.predictions) = self.create_model(q, optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, q, optimizer):
        """Model function for ResNet-18 on CIFAR-10."""
        features = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='features')
        labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

        def conv_bn_relu(inputs, filters, kernel_size, strides, name):
            with tf.variable_scope(name):
                conv = tf.layers.conv2d(
                    inputs,
                    filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=tf.variance_scaling_initializer())
                bn = tf.layers.batch_normalization(conv, training=is_training)
                return tf.nn.relu(bn)

        def residual_block(inputs, filters, strides, name):
            with tf.variable_scope(name):
                shortcut = inputs
                conv1 = conv_bn_relu(inputs, filters, 3, strides, name='conv1')
                conv2 = tf.layers.conv2d(
                    conv1,
                    filters,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=tf.variance_scaling_initializer(),
                    name='conv2')
                conv2 = tf.layers.batch_normalization(conv2, training=is_training, name='bn2')
                input_channels = inputs.get_shape().as_list()[-1]
                if strides != 1 or input_channels != filters:
                    shortcut = tf.layers.conv2d(
                        inputs,
                        filters,
                        kernel_size=1,
                        strides=strides,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=tf.variance_scaling_initializer(),
                        name='shortcut_conv')
                    shortcut = tf.layers.batch_normalization(shortcut, training=is_training, name='shortcut_bn')
                output = tf.nn.relu(conv2 + shortcut)
                return output

        net = conv_bn_relu(features, 64, 3, 1, name='stem')

        for block_idx in range(2):
            net = residual_block(net, 64, 1, name='block1_{}'.format(block_idx))
        net = residual_block(net, 128, 2, name='block2_0')
        net = residual_block(net, 128, 1, name='block2_1')
        net = residual_block(net, 256, 2, name='block3_0')
        net = residual_block(net, 256, 1, name='block3_1')
        net = residual_block(net, 512, 2, name='block4_0')
        net = residual_block(net, 512, 1, name='block4_1')

        net = tf.reduce_mean(net, axis=[1, 2], name='global_avg_pool')
        logits = tf.layers.dense(net, self.num_classes, name='fc')

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions))
        return features, labels, is_training, train_op, grads, eval_metric_ops, loss, predictions

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                assign_ops = []
                for variable, value in zip(all_vars, model_params):
                    assign_ops.append(variable.assign(value))
                self.sess.run(assign_ops)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={
                    self.features: data['x'],
                    self.labels: data['y'],
                    self.is_training: True
                })
            grads = process_grad(model_grads)

        return num_samples, grads

    def get_loss(self, data):
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={
                self.features: data['x'],
                self.labels: data['y'],
                self.is_training: False
            })
        return loss

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        for _ in range(num_epochs):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: X, self.labels: y, self.is_training: True})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                    feed_dict={
                                        self.features: mini_batch_data[0],
                                        self.labels: mini_batch_data[1],
                                        self.is_training: True
                                    })

        weights = self.get_params()
        return grads, loss, weights

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                feed_dict={
                    self.features: data['x'],
                    self.labels: data['y'],
                    self.is_training: False
                })
        return tot_correct, loss

    def close(self):
        self.sess.close()
