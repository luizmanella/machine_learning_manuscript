import tensorflow as tf
from tensorflow import keras


# Load Fashion-MNIST data from Keras data repository
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Fashion-MNIST like traiditional MNIST are 28x28
IMG_SHAPE = (28, 28, 1)

# Batch size can vary depending on the memory availability.
# Number of epochs and batch can obviously be updated to
# improve results based on computer power
BATCH_SIZE = 32
epochs = 10

# size of noise vector
noise_dim = 128

# Add new axis to the image - turns image from 28x28 to 28x28x1
train_images = train_images[..., tf.newaxis].astype('float32')

# Normalize pixels to [-1, 1]
train_images = (train_images - 127.5) / 127.5


def conv_block(x, filters, use_bn=False, use_dropout=False):
    """
    :description:
        This is a convolutional block. We use it to create the discriminator
        later and to keep the code simpler to read.

    :param x: (tf) discriminator model
    :param filters: (int) filter size
    :param use_bn: (boolean) applies batch normalization
    :param use_dropout: (boolean) applies dropout
    :return: (tf) model with additional convolutional blocks
    """
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    if use_dropout:
        x = tf.keras.layers.Dropout(0.5)(x)
    return x


def get_discriminator_model():
    """
    :description:
        This function creates and returns the discriminator.
        It is a 4 layer convolutional network ending with a
        dense layer to compress output to 1 value.

    :return: (tf) discriminator model
    """
    img_input = tf.keras.layers.Input(shape=IMG_SHAPE)
    # Add zero padding. If we don't then we will have a size problem when downsampling/upsampling.
    x = tf.keras.layers.ZeroPadding2D((2, 2))(img_input)

    # Convolutional layers
    x = conv_block(x, 64, use_bn=False, use_dropout=False)
    x = conv_block(x, 128, use_bn=False, use_dropout=True)
    x = conv_block(x, 256, use_bn=False, use_dropout=True)
    x = conv_block(x, 512, use_bn=False, use_dropout=False)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    d_model = tf.keras.models.Model(img_input, x, name="discriminator")

    return d_model


def upsample_block(x, filters, activation, use_bn=False, use_bias=True, use_dropout=False):
    """
    :description:
        This function is called to up-sample images in the generator.

    :param x: (tf) model
    :param filters: (int) filter size
    :param activation: (tf) activation function
    :param use_bn: (boolean) used to apply batch normalization
    :param use_bias: (boolean) used to include bias
    :param use_dropout: (boolean) used to apply dropout
    :return: (tf) updated model
    """
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', use_bias=use_bias)(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    if use_dropout:
        x = tf.keras.Dropout(0.3)(x)
    return x


def get_generator_model():
    """
    :description:
        This function creates the generator model for our WGAN.

    :return: (tf) generator model
    """
    noise = tf.keras.layers.Input(shape=(noise_dim))
    x = tf.keras.layers.Dense(4 * 4 * 256, use_bias=False)(noise)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Reshape((4, 4, 256))(x)
    x = upsample_block(x, 128, tf.keras.layers.LeakyReLU(0.2), use_bn=True, use_bias=False)
    x = upsample_block(x, 64, tf.keras.layers.LeakyReLU(0.2), use_bn=True, use_bias=False)
    x = upsample_block(x, 1, tf.keras.layers.Activation("tanh"), use_bn=True, use_bias=False)

    # We need to remove the padding that we added prior to return the image to a 28x28
    x = tf.keras.layers.Cropping2D((2, 2))(x)

    g_model = tf.keras.models.Model(noise, x, name='generator')
    return g_model


def discriminator_loss(real_img, fake_img):
    """
    :description:
        Loss function for the discriminator.

    :param real_img: tensor) of real image image
    :param fake_img: (tensor) generated image
    :return: (int) loss
    """
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    """
    :description:
        Function to compute the generator loss.

    :param fake_img:
    :return:
    """
    return -tf.reduce_mean(fake_img)



class WGAN(tf.keras.Model):
    """
    :description:
        This is the WGAN class which will handle everything regarding the
        creation and training of the WGAN model.

    :process:
        For each batch, we are going to perform the following steps as
        laid out in the original paper:
        1. Train the discriminator and get its loss
            - As described in the paper we train 5x more than the generator
        2. Train the generator and get its loss
        3. Calculate the gradient penalty and multiply with a weight factor
        4. Add the gradient penalty to the discriminator loss
    """
    def __init__(self, discriminator, generator, latent_dim, discriminator_extra_steps=5, gp_weight=10.0):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)

                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)

                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)

                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)

            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # -----------------------------------------
        #             Train the generator
        # -----------------------------------------
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)

            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)

            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)

        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}



# ------------------------------------------------------------------------------
# 							TRAIN WGAN MODEL
# ------------------------------------------------------------------------------


# discriminator
d_model = get_discriminator_model()

# generator
g_model = get_generator_model()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

# Create the WGAN model.
wgan = WGAN(discriminator=d_model,
            generator=g_model,
            latent_dim=noise_dim,
            discriminator_extra_steps=5
)
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss
)
wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=epochs)


# ------------------------------------------------------------------------------
# 							STORE GENERATED IMAGES
# ------------------------------------------------------------------------------
# To generate images we sample from a normal distribution and pass that sample
# to the generator.
latent_vector = tf.random.normal(shape=(6, 128))
generated_images = wgan.generator(latent_vector)

# We undo the normalization process here
generated_images = (generated_images * 127.5) + 127.5
for i in range(5):
    img = generated_images[i].numpy()
    img = keras.preprocessing.image.array_to_img(img)

    # Path expects a folder called imgs to keep things neat and separated
    img.save(f'./imgs/fake_img_{i}.png')

