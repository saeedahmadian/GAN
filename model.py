import tensorflow as tf
import numpy as np

class CyberGAN(object):
    def __init__(self, config):

        self.num_samples = config.num_samples
        self.num_features = config.num_features
        self.num_nodes = config.num_nodes

        self.batch_size = config.batch_size
        self.epoch = config.epoch

        # Inputs
        self.gen_input = tf.placeholder(tf.float32, shape=[None, self.num_nodes], name='gen_input')
        self.disc_input = tf.placeholder(tf.float32, shape=[None, self.num_nodes, self.num_features, 1],
                                         name='disc_input')

        # Targets (Real input: 1, Fake input: 0)
        self.disc_target = tf.placeholder(tf.float32, shape=[None, 1], name='disc_target')
        self.gen_target = tf.placeholder(tf.float32, shape=[None, 1], name='gen_target')

        self.build_model()

    def build_model(self):

        # Build generator
        gen_out = self.generator(self.gen_input)

        # Build 2 Discriminator Networks (one from noise input, one from generated samples)
        disc_out_real = self.discriminator(self.disc_input)
        disc_out_fake = self.discriminator(gen_out, reuse=True)
        disc_concat = tf.concat([disc_out_real, disc_out_fake], axis=0)
        disc_concat_target = tf.concat([self.disc_target, self.gen_target], axis=0)

        # Build the stacked generator/discriminator
        stacked_out = self.discriminator(gen_out, reuse=True)

        # Build Loss 1
        # Discriminator tries to discriminate real or fake input
        self.disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_concat, labels=disc_concat_target))
        # Generator tries to fool discriminator => label=1
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=stacked_out, labels=tf.ones_like(self.gen_target)))


        # Build Optimizers
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.001)

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        # Discriminator Network Variables
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        # Create training operations
        self.train_gen = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def generator(self, z):
        with tf.variable_scope("generator"):

            h1 = tf.layers.dense(z, units=self.num_nodes * self.num_features // 2,
                                 activation=tf.nn.relu, name='h1_dense')
            h1 = tf.reshape(h1, shape=[-1, self.num_nodes, self.num_features // 2, 1], name='h1_reshape')
            h2 = tf.layers.conv2d_transpose(inputs=h1, filters=1, kernel_size=(1, 2), strides=(1, 2),
                                            padding='same', activation=tf.nn.relu, name='h2')
            return h2

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            h1 = tf.layers.conv2d(x, 10, kernel_size=[1, 3], strides=[1, 2], activation=tf.nn.relu)
            h2 = tf.layers.conv2d(h1, 50, kernel_size=[1, 3], strides=[1, 2], activation=tf.nn.relu)
            h3 = tf.layers.average_pooling2d(h2, 1, 2)
            h3 = tf.contrib.layers.flatten(h3)
            h4 = tf.layers.dense(h3, 50)
            h4 = tf.nn.relu(h4)
            out = tf.layers.dense(h4, 1)

            return out

    def train(self, data):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(self.init)

            step = 0
            for i in range(self.epoch):
                data.randomize()
                for batch_x, batch_y in data.next_batch(self.batch_size):

                    # Generate noise to feed to the generator
                    z_sample = np.random.uniform(0., 1., size=[batch_x.shape[0], self.num_nodes]).astype('float32')
                    z_label = np.ones_like(batch_y, dtype='float32')
                    # Train
                    feed_dict = {self.disc_input: batch_x, self.gen_input: z_sample,
                                 self.disc_target: batch_y, self.gen_target: z_label}
                    _, _, g_loss, d_loss = sess.run([self.train_gen, self.train_disc, self.gen_loss, self.disc_loss],
                                                    feed_dict=feed_dict)

                    if step % 100 == 0:
                        print('Step {}: Generator Loss: {:.2f}, Discriminator Loss: {:.2f}'.format(step, g_loss, d_loss))
                    step += 1