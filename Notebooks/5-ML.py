import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <div style="color:brown;font-family:arial;font-size:26pt;font-weight:bold;text-align:center"> 
    Example of a marimo notebook for Machine Learning</div>
    <hr>
    <div style="color:blue;font-family:arial;font-size:22pt;font-weight:bold;text-align:center"> 
    Training a Dense Neural Network to classify<br>handwritten digits from MINIST database</div>
    """
    )
    return


@app.cell
def _():
    import os, sys
    from pathlib import Path

    # Delete the (numerous) warning messages from the **tensorflow** module:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time

    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split

    print(f"Python    : {sys.version.split()[0]}")
    print(f"tensorflow: {tf.__version__} with {keras.__version__}")
    print(f"numpy     : {np.__version__}")

    # local module
    from utils.tools import (scan_dir, plot_images, plot_loss_accuracy, elapsed_time_since, 
                             show_conf_matrix, plot_proportion_bar)

    SEED = 1234
    return (
        SEED,
        elapsed_time_since,
        keras,
        plot_images,
        plot_loss_accuracy,
        plot_proportion_bar,
        show_conf_matrix,
        tf,
        time,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1 - Load the MNIST data (images and labels)""")
    return


@app.cell
def _(keras):
    (image_train, label_train), (image_valid, label_valid) = keras.datasets.mnist.load_data()
    return image_train, image_valid, label_train, label_valid


@app.cell
def _(image_train, image_valid, label_train, label_valid):
    print(f"{image_train.shape=}, {image_train.dtype=}")
    print(f"{label_train.shape=}, {label_train.dtype=}")
    print(f"{image_valid.shape=}, {image_valid.dtype=}")
    print(f"{label_valid.shape=}, {label_valid.dtype=}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### ▸ Images and labels visualisation:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""With the `plot_image` function (from the `utils.tools` module) we can display images and labels starting with image #600 on a grid of 4 x 12:""")
    return


@app.cell
def _(image_train, label_train, plot_images):
    plot_images(image_train, 4, 12, 599, label_array=label_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### ▸ Create the 3 datasets: train, validation & test

    To follow the _state of the art_, we will split the dataset into **train**, **validation** & **test** datasets.<br>
    A simple way to do this is to keep the train dataset and to split the current validation dataset in two equal subsets:<br>
    - a smaller validation subset<br>
    - a test subset.
    """
    )
    return


@app.cell
def _(SEED, image_valid, label_valid, train_test_split):
    # note on 'train_test_split' from sklearn module : Stratified train/test split is not implemented for shuffle=False 
    # We give the seed value with the 'random_state' parameter to ensure a reproducible splitting.

    img_valid, img_test, lab_valid, lab_test = train_test_split(image_valid, label_valid,
                                                                stratify=label_valid,
                                                                test_size=0.5,
                                                                shuffle=True,
                                                                random_state=SEED)
    return img_test, img_valid, lab_test, lab_valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(image_train, img_test, img_valid, label_train):
    # Rename the train dataset components:
    img_train, lab_train = image_train, label_train

    # Let's check the sizes of the 3 datasets:
    print(f'train:  {img_train.shape}')
    print(f'valid:  {img_valid.shape}')
    print(f'test :  {img_test.shape}')
    return img_train, lab_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now let's verify that the proportion of digits remains homogenous through the new `valid` & `test` datasets:""")
    return


@app.cell
def _(lab_test, lab_valid, plot_proportion_bar):
    prop = {}
    prop['valid'] = [ (lab_valid == i).sum() for i in range(10)]
    prop['test']  = [ (lab_test  == i).sum() for i in range(10)]
    plot_proportion_bar(prop, range(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// Admonition |  Define useful parameters

    To avoid *hard writing* the **number of training and test images**, the **dimension** of the images and the **number of classes** to recognize, these parameters are retrieved from the dataset:<br>
    ▸ the `shape` attribute of the `im_train` and `im_test` arrays includes the number of training and test images<br>
    ▸ the `size` attribute of any training or test image gives the number of pixels of the images (784),<br>
    ▸ the transformation of the `lab_test` array into a Python `set` (a set) gives the set of labels to recognize, whose size is the number of classes.
    ///
    """
    )
    return


@app.cell
def _(img_test, img_train, img_valid, lab_train):
    # number of images in the 3 sets:
    NB_IMG_TRAIN = img_train.shape[0]
    NB_IMG_VALID = img_valid.shape[0]     
    NB_IMG_TEST  = img_test.shape[0]   

    # number of pixels:
    NB_PIXEL = img_train[0].size

    # number of classes:
    NB_CLASS = len(set(lab_train))

    # image shape:
    IMG_SHAPE = img_train[0].shape

    # check:
    print(f'{NB_IMG_TRAIN=}\n{NB_IMG_VALID=}\n{NB_IMG_TEST=}\n{NB_PIXEL=}')
    print(f'{NB_CLASS} different classes found in `lab_train` array')
    print(f'Image size: {IMG_SHAPE}')
    return NB_CLASS, NB_IMG_TEST, NB_IMG_TRAIN, NB_IMG_VALID, NB_PIXEL


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2 - Pre-process MNIST images and labels

    Two treatments are applied to the MNIST datasets:<br>
    ▸ image arrays: transform the 28$\,\times\,$28 pixels (`uint8` integers) arrays into **normalized** vectors $(V_i)_{i=0..783}$ of 784 `float` values $V_i$ with $0 \leqslant V_i \leqslant 1$<br>
    ▸ labels: encode the `int` numbers into *one-hot* vectors.<br>
    We define `x_train`, `y_train` and `x_valid`, `y_valid` the new train & valid datasets.
    """
    )
    return


@app.cell
def _(
    NB_IMG_TEST,
    NB_IMG_TRAIN,
    NB_IMG_VALID,
    NB_PIXEL,
    img_test,
    img_train,
    img_valid,
):
    x_train = img_train.reshape(NB_IMG_TRAIN, NB_PIXEL)/255
    x_valid = img_valid.reshape(NB_IMG_VALID, NB_PIXEL)/255
    x_test  = img_test.reshape(NB_IMG_TEST, NB_PIXEL)/255

    #check:
    print(f'train: {x_train.shape}, min: {x_train.min()}, max: {x_train.max()}')
    print(f'valid: {x_valid.shape}, min: {x_valid.min()}, max: {x_valid.max()}')
    print(f'test : {x_test.shape}, min: {x_test.min()}, max: {x_test.max()}')
    return x_test, x_train, x_valid


@app.cell
def _(lab_test, lab_train, lab_valid):
    from tensorflow.keras.utils import to_categorical
    # 'one-hot' encoding' of labels :
    y_train = to_categorical(lab_train)
    y_valid = to_categorical(lab_valid)
    y_test  = to_categorical(lab_test)
    return y_test, y_train, y_valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3 - Build the Dense Neural Network""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We build a **dense network**, with:<br>
    - an __input layer__ of 784 values in the range [0 ; 1.]<br>
    - a **hidden layer** of 784 neurons using the `relu` activation function,<br>
    - an **output layer** with 10 neurons, for the classification of images into the 10 digits (0,1,2...9), using the `softmax` activation function adapted to classification problems .

    <img src="public/ReseauChiffres-2_transp.png" width="95%" />
    """
    )
    return


@app.cell
def _(tf):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input

    def build_DNN(nb_input, nb_neuron, nb_class, seed=None):

        if seed is not None:
            ##########################
            # Deterministic training #
            ##########################
            # 1/ set the seed of the random generators involved by tensorflow:
            tf.keras.utils.set_random_seed(seed)
            # 2/ make the tf ops determinisctic 
            # [see https://blog.tensorflow.org/2022/05/whats-new-in-tensorflow-29.html]
            tf.config.experimental.enable_op_determinism() 

        model = Sequential()
        model.add(Input(shape=(nb_input,), name='input'))             # INPUT layer
        model.add(Dense(nb_neuron, activation='relu', name='c1'))     # First hidden layer
        model.add(Dense(nb_class, activation='softmax', name='c2'))   # OUTPUT layer
        model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
        return model

    return (build_DNN,)


@app.cell
def _(NB_CLASS, NB_PIXEL, SEED, build_DNN):
    model = build_DNN(nb_input=NB_PIXEL, 
                      nb_neuron=NB_PIXEL, 
                      nb_class=NB_CLASS, 
                      seed=SEED)
    model.summary()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4 - Train the model

    Let's define the `train_model` function:<br>
    ▸ Trains the model with a *training dataset* given as argument.<br>
    ▸ Evaluates the model after each *epoch* with a *validation dataset* given as argument.<br>
    ▸ Monitors the *evaluation loss* metric to do *early-stopping*.<br>
    ▸ Uses the hyper-parameters `epochs` & `batch_size` for the training.
    """
    )
    return


@app.cell
def _(elapsed_time_since, time, x_valid, y_valid):
    from tensorflow.keras.callbacks import EarlyStopping

    def train(model, xtrain, ytrain, patience=2, n_epoch=10, b_size=32):
        '''
        Train the model using the datsets ans the hyperparameters given as arguments.
        '''

        callbacks_list = [
            EarlyStopping(monitor='val_loss',  # The parameter to monitor
                          patience=patience,   # accept that 'val_loss' decrease 1 time
                          restore_best_weights=True,
                          verbose=1)
        ]

        t0 = time()
        hist = model.fit(xtrain, ytrain,       # images, labels
                         epochs=n_epoch,       # the maximal number of successive trainings
                         batch_size=b_size,    # fragmentation of the whole dada set in batches
                         validation_data=(x_valid, y_valid), 
                         verbose=1,
                         callbacks = callbacks_list)

        print(f' Total Train {elapsed_time_since(t0)}')   

        return model, hist
    return (train,)


@app.cell
def _(model, train, x_train, y_train):
    hyp_param = {'patience':2, 'n_epoch':10, 'b_size':128}
    trained_model, hist = train(model, x_train, y_train, **hyp_param)
    return hist, trained_model


@app.cell
def _(hist, plot_loss_accuracy):
    plot_loss_accuracy(hist)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4 - Evaluate the trained model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The accuracy and loss of the trained model is evaluated with the **test** dataset:""")
    return


@app.cell
def _(trained_model, x_test, x_valid, y_test):
    test_loss, test_acc = trained_model.evaluate(x_test, y_test, verbose=0)
    results = trained_model.predict(x_valid, verbose=0)
    inferences = results.argmax(axis=-1)

    f'{test_acc=:.3f}, {test_loss=:.3f}'
    return (inferences,)


@app.cell
def _(inferences, lab_valid, show_conf_matrix):
    show_conf_matrix(lab_valid, inferences, cmap='magma', classes=range(10), figsize=(7,6))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
