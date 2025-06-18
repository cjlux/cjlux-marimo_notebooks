import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <div style="color:brown;font-family:arial;font-size:28pt;font-weight:bold;text-align:center"> 
    Example of a marimo notebook for Machine Learning</div>
    <hr>
    <div style="color:blue;font-family:arial;font-size:22pt;font-weight:bold;text-align:center"> 
    Training a Dense Neural Network to classify<br>handwritten digits from MINIST database</div>
    """
    )
    return


@app.cell
def _():
    import os, sys, io, contextlib, pickle
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
        Path,
        SEED,
        contextlib,
        elapsed_time_since,
        io,
        keras,
        np,
        os,
        plot_images,
        plot_loss_accuracy,
        plot_proportion_bar,
        scan_dir,
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
    mo.md(
        r"""
    ## 3 - Build the Dense Neural Network

    The Dense Neural Network can be build with 5 lines thanks to the **keras** module:
    """
    )
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
        '''
        Build a Dense Neural Network with the given arguments 
        '''

        if seed is not None:
            ############################
            # Deterministic processing #
            ############################
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4 - Save the structure & weights of the DNN

    The `save` method of the `Sequential` class saves **the structure** and the **weights** of the DNN.<br>
    We can use later the `tf.keras.models.load_model` function to recreate the network and reload its weights.
    """
    )
    return


@app.cell
def _(Path, scan_dir):
    def save_model(model, file_name, list_dir=False):
        '''
        Save the model structure & weights in a file.
        '''
        # Create the directory 'model' if needed:
        model_path = Path("models")
        model_path.mkdir(exist_ok=True)
        # Check the fle name:
        if not file_name.endswith('.keras'): file_name += '.keras'
        model.save(model_path / file_name)
        # display the tree beginning at f'models/':
        if list_dir:
            tree = scan_dir(model_path)
            print(f'\nFiles in directory <{model_path}>:\n{tree}')    
    return (save_model,)


@app.cell
def _(NB_CLASS, NB_PIXEL, SEED, build_DNN, save_model):
    # build a new network/
    _model = build_DNN(nb_input=NB_PIXEL, nb_neuron=NB_PIXEL, nb_class=NB_CLASS, seed=SEED)

    # save the initial model:
    save_model(_model, 'dense1_init.keras', list_dir=True)
    return


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

    def train(model, xtrain, ytrain, xvalid, yvalid, patience=2, n_epoch=10, b_size=32):
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
                         verbose=2,
                         callbacks = callbacks_list)

        print(f' Total Train {elapsed_time_since(t0)}')   

        return model, hist
    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// Warning | CPU-hungry cell 

    **The next cell runs the model training which can be very expensive in CPU time...**

    $\leadsto$ We use the `mo.persistent_cache` marimo context to avoid unnecessary execution of the cell: objets defined in this cell are saved in a binary file on disk in order to be retrieved when necessary.<br>
    For example:<br>
    ▸ At notebook startup, if the notebook has already been executed once, all the objetcs of this cell are retrieved from the cache disk.<br>
    ▸ If one re-run teh cell without modification, no need to do anything.
    ///
    """
    )
    return


@app.cell
def _(
    Path,
    SEED,
    Tee,
    contextlib,
    mo,
    np,
    tf,
    train,
    x_train,
    x_valid,
    y_train,
    y_valid,
):
    with mo.persistent_cache(name="DNN_cache"):
        '''
        This block of code and its computed variables will be cached to disk
        the first time it's run. The next time it's run, objects will be loaded from disk.
        '''

        # load the network initial structure and weights:
        model_init = tf.keras.models.load_model(Path('models') / 'dense1_init.keras') 

        # Deterministic tensorflow training: 
        tf.keras.utils.set_random_seed(SEED)
        # see https://blog.tensorflow.org/2022/05/whats-new-in-tensorflow-29.html
        tf.config.experimental.enable_op_determinism() 

        # train the DNN:
        hyp_param ={'patience':1, 'n_epoch':10, 'b_size':32}
        # Create an identifier string for the trained model like 'patience_2-n_epochs_10-b_size_32.keras':
        model_id = '-'.join(['_'.join((k, str(hyp_param[k]))) for k in hyp_param.keys()])

        tee_stdout = Tee() 
        with contextlib.redirect_stdout(tee_stdout):
            trained_model, hist = train(model_init, x_train, y_train, x_valid, y_valid, **hyp_param)

        # Get the captured stdout content as a string
        training_output = tee_stdout.string_io.getvalue()

        # save the output on disk
        log_file = model_id + ".log"
        log_path = Path("models") / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)  
        log_path.write_text(training_output)

        # save the history on disk:
        hist_file = model_id + ".hist"
        hist_path = Path("models") / hist_file
        np.save(hist_path, hist)

        # Save the model structure and weights to disk
        save_file_name = model_id + '.keras'
        save_file_path = Path('models/') / save_file_name
        trained_model.save(save_file_path)

    return hist, log_file, model_id, save_file_name


@app.cell(hide_code=True)
def _(mo):
    see_log = mo.ui.checkbox(label="See current log file")
    see_log
    return (see_log,)


@app.cell(hide_code=True)
def _(Path, cell_output, log_file, see_log):
    if see_log.value:
        log_file_Path = Path("models") / log_file
        cell_output(log_file_Path) 
    return


@app.cell
def _(hist, model_id, plot_loss_accuracy):
    plot_loss_accuracy(hist, message=model_id, figsize=(10,5)) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5 - Evaluate a trained model""")
    return


@app.cell(hide_code=True)
def _(mo, os, save_file_name):
    _options = [f for f in os.listdir('./models') if f.endswith('.keras')]
    _options.sort()
    choose_model = mo.ui.dropdown(options=_options, label="Choose saved trained model",
                                value=save_file_name)
    choose_model
    return (choose_model,)


@app.cell(hide_code=True)
def _(Path, choose_model, mo, tf, x_test, x_valid, y_test):
    model = tf.keras.models.load_model(Path('models') / choose_model.value) 
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    results = model.predict(x_valid, verbose=0)
    inferences = results.argmax(axis=-1)

    mo.plain_text(f'{test_acc=:.3f}, {test_loss=:.3f}')
    return (inferences,)


@app.cell(hide_code=True)
def _(inferences, lab_valid, show_conf_matrix):
    show_conf_matrix(lab_valid, inferences, cmap='magma', classes=range(10), figsize=(7,6))
    return


@app.cell(hide_code=True)
def _(Path, choose_model, np, plot_loss_accuracy):
    _model_id = choose_model.value.replace('.keras','')
    _hist_path = Path("models") / (_model_id + '.hist.npy')
    if _hist_path.exists():
        _hist = np.load(_hist_path, allow_pickle='TRUE').item()
    plot_loss_accuracy(_hist, message=_model_id, figsize=(10,5)) if _hist_path.exists() else None
    return


@app.cell(hide_code=True)
def _(Path, cell_output, choose_model):
    _log_file_Path = Path("models") / choose_model.value.replace('.keras', '.log')
    cell_output(_log_file_Path) 
    return


@app.cell
def _(io, mo):
    class Tee(io.TextIOBase):
        '''
        To write model.fit() stdout both on the call output and in a string that can be written in an ASCII file.
        '''
        def __init__(self):
            self.string_io = io.StringIO()

        def write(self, data):
            self.string_io.write(data)
            mo.output.append(data)
            return len(data)

        def flush(self):
            self.string_io.flush()

    return (Tee,)


@app.cell
def _(mo):
    def cell_output(file_Path):
        if file_Path.exists():
            mo.output.append(f"Log for model <{file_Path.name.replace('.log','')}>:")
            mo.output.append(mo.Html(file_Path.read_text().replace('\n', '<br>')))
    return (cell_output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <iframe src="https://www.youtube.com/embed/aircAruvnKk" width="400" height="300" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    <br>
    <iframe src="https://www.youtube.com/embed/IHZwWFHWa-w"width="400" height="300" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    <br>
    <iframe src="https://www.youtube.com/embed/Ilg3gGewQ5U" width="400" height="300" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
