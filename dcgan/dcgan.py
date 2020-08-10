from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import pathlib

from matplotlib import pyplot
from keract import get_activations, display_activations, display_heatmaps



def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same', name='Gen_64'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same', name='Gen_1'))
    model.add(Activation('tanh'))
    return model



def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5),
                    padding='same',
                    input_shape=(28, 28, 1),
                    name = "discrim_64"
                    )
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5),
                    name='Discrim_128'
                    )
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model



def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model



def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image



def train(BATCH_SIZE, EPOCHS, WARM_START=False):
    epochs_start = 0
    pathlib.Path('./images').mkdir(exist_ok=True)
    #get training data
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]

    #instantiate the models
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)

    #if we want a warm start, load in the model weights from the file
    if WARM_START:
        #make sure to start the epochs count where we left off
        #so not to delete photos!
        epochs_start = EPOCHS
        EPOCHS = EPOCHS * 2
        #then get the weights out of the files
        g.load_weights('generator')
        d.load_weights('discriminator')

    #compile the models
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    #now enter the training loop
    for epoch in range(epochs_start, EPOCHS):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        #within each epoch, train on # images/batch size
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            #generate random noise for generator to turn into image
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            #get batch of images from MNIST
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            #make generator generate image
            generated_images = g.predict(noise, verbose=0)
            #every 20 batches, make a grid of generated images to output
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                image_path = "./images/{}_{}.png".format(epoch, index)
                Image.fromarray(image.astype(np.uint8)).save(image_path)
            #concatenate half real MNIST + generated images
            #along with labels [1=True, 0=False] to train discriminator on
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            #make some more noise for the generator
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            #set the discriminator to un-trainable
            d.trainable = False
            #train the paired model on noise
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            #print stats
            print("Batch {}   d_loss: {:.2f}    g_loss: {:.2f}".format(index, d_loss, g_loss))
            #every 10 steps, save the weights
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)



def generate(BATCH_SIZE, nice=False):
    #load model weights from files
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')

    d = discriminator_model()
    d.compile(loss='binary_crossentropy', optimizer="SGD")
    d.load_weights('discriminator')

    #if we want to see the nicest photos
    if nice:
        #make the discriminator pick some out for us
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    #otherwise just generate some regardless of how they look
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")



def activation_fun():
    #load the model weights in from  files
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    d = discriminator_model()
    d.compile(loss='binary_crossentropy', optimizer="SGD")
    d.load_weights('discriminator')

    #generate the pics
    noise = np.random.uniform(-1, 1, (1, 100))
    generated_images = g.predict(noise, verbose=1)

    #get the activations for the generator
    g_acti_1 = get_activations(g, noise, layer_name='Gen_1')
    display_activations(g_acti_1, save=True, directory='.')
    #get the activations for the generator
    g_acti_64 = get_activations(g, noise, layer_name='Gen_64')
    display_activations(g_acti_64, save=True, directory='.')


    #get the activations for the discriminator
    d_acti_128 = get_activations(d, generated_images, layer_name='Discrim_128')
    display_activations(d_acti_128, save=True, directory='.')
    #get the activations for the discriminator
    d_acti_64 = get_activations(d, generated_images, layer_name='discrim_64')
    display_activations(d_acti_64, save=True, directory='.')




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warm_start", type=bool, default=False )
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, EPOCHS=args.epochs, WARM_START=args.warm_start)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
    elif args.mode == "activations":
        activation_fun()
