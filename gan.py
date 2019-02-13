import tensorflow as tf
# tf.enable_eager_execution()

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

from IPython import display

# experiment parameters
EPOCHS = 10
NOISE_DIM = 100
BATCH_SIZE = 256


def load_mnist():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()


    train_images = train_images.reshape(train_images.shape[0],
                                        28, 28, 1).astype('float32')
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    BUFFER_SIZE = 60000
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
      
    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
    
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)  
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
  
    return model
def make_discriminator_model():
    model = tf.keras.Sequential()
    # I'm add the shape here
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(28,28,1)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
      
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
       
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
     
    return model

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

def step(images, generator, discriminator, generator_optimizer,
         discriminator_optimizer):
    # generating noise from a normal distribution
    noise = tf.random_normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    g_of_generator = gen_tape.gradient(gen_loss,
                                       generator.variables)
    g_of_discriminator = disc_tape.gradient(disc_loss,
                                            discriminator.variables)

    generator_optimizer.apply_gradients(zip(g_of_generator,
                                            generator.variables))
    discriminator_optimizer.apply_gradients(zip(g_of_discriminator,
                                                discriminator.variables))
# step = tf.contrib.eager.defun(step)

def train_gan(dataset, generator, discriminator, generator_optimizer,
              discriminator_optimizer):
    for epoch in range(EPOCHS):
        iterator = train_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        start = time.time()
        for n,images in enumerate(dataset.take(10)):
            step(images, generator, discriminator,
                 generator_optimizer, discriminator_optimizer)
        print ('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                          time.time()-start))


def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))
  
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()

def main():
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    generator_optimizer = tf.train.AdamOptimizer(1e-4)
    discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

    discriminator.summary()
    generator.summary()
    config = tf.ConfigProto(allow_soft_placement=True,
                            gpu_options.allow_growth=True)
    sess = tf.Session(config=config)

    train_dataset = load_mnist()
    # while True:
    #     print('.')
    #     sess.run(next_element)
    train_dataset
    sess.run(train_dataset)
    # tf.zeros(8).eval()
    with tf.Session() as sess:
        print(sess.run(train_dataset.take(1)))
        print(sess.run(tf.zeros(8)))
    next_element.shape
    res = sess.run(next_element)
    res.shape
    train_gan(train_dataset, generator, discriminator,
              generator_optimizer, discriminator_optimizer)
    x = tf.placeholder(tf.float32, shape=[3])
def test_dataset():
    dataset = tf.data.Dataset.range(100)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    for i in range(100):
      value = sess.run(next_element)
      assert i == value
def test_checkpoint():
    # checkpoint save and load
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    checkpoint.save(file_prefix = checkpoint_prefix)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def test():
    num_examples_to_generate = 16
    # We'll re-use this random vector used to seed the generator so it
    # will be easier to see the improvement over time.
    random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                     NOISE_DIM])
    generate_and_save_images(generator, 9999, random_vector_for_generation)

    mynoise = tf.random_normal([10, NOISE_DIM])
    my_generated_images = generator(mynoise, training=False)
    my_generated_images.shape
    discriminator.summary()
    discriminator(my_generated_images[:10], training=False)
    train_images.shape
    discriminator(train_images[:10], training=False)

def test():
    x1 = tf.ones(shape=(10,5))
    x2 = tf.ones(shape=(10,5))
    x3 = tf.zeros(shape=(10,5))
    x1
    x1 = tf.convert_to_tensor([1,1,1])
    x2 = tf.convert_to_tensor([1,1,1])

    x2 = tf.add(x1, x2)
    tf.negative(x1)
    x1
    # cross entropy
    tf.losses.sigmoid_cross_entropy(x1, x2)
    
    tf.losses.sigmoid_cross_entropy(x1, tf.negative(x2))
    
    tf.losses.sigmoid_cross_entropy(x1, x1)
    tf.losses.sigmoid_cross_entropy(x1, x3)
    tf.losses.softmax_cross_entropy(x1, x2)
    tf.losses.softmax_cross_entropy(x1, x2)
    # tf.one_hot([1,2,3], 8)
    # other distance measure
    tf.losses.cosine_distance(x1, x2, axis=0)
    tf.losses.absolute_difference(x1, x2)
    tf.losses.hinge_loss(x1, x2)
    tf.losses.hinge_loss(x1, x3)
    tf.losses.mean_squared_error(x1, x2)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(x2 * tf.log(x1), reduction_indices=[1]))
    cross_entropy

