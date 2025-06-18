# version 2.2 - 2025/03/06

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from seaborn import heatmap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from time import time
from cpuinfo import get_cpu_info

def display_history_stat(hist:list):
    metrics = ('accuracy', 'loss', 'val_accuracy', 'val_loss')
    for m in metrics:
        values = np.array([h.history[m] for h in hist])
        print(f'{m}: mean:{values.mean():.3f} std:{values.std():.3f} ', end='')

def elapsed_time_since(t0):
    t = int(time()-t0)
    h = int(t//3600)
    m = int((t - h*3600)//60)
    s = int((t - h*3600 -m*60))
    return f"Elapsed time {t}s -> {h:02d}:{m:02d}:{s:02d}"
    
def plot_proportion_bar(proportions:dict, class_names, figsize=(6,4), plt_show=False):
    '''
    To plot propotion of classes in different datasets.
    proportion: the dictionnary {<dataset name>: <[number of class in teh dataset]>}
    '''
    width = 0.45
    coeff = 0
    x = np.arange(len(class_names))  # the label locations on x axis
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    for name, values in proportions.items():
        offset = width * coeff
        rects = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(rects, padding=3, fontsize=8)
        coeff += 1
    ax.set_ylabel('number of digits')
    ax.set_xlabel('Classes')
    ax.set_title('Proportion of digits in dataset')
    ax.set_xticks(x + width/2, class_names)
    ax.legend()
    if plt_show: plt.show()
    return fig

def split_stratified_into_train_val_test(dataset: (ndarray, ndarray), 
                                         frac_train:float=0.7, 
                                         frac_val:float=0.15,
                                         frac_test:float=0.15, 
                                         shuffle:bool=False,
                                         seed:float=None): 
    '''
    Splits a numpy dataset (data, labesl) into three subsets train, val, and test
    following fractional ratios provided, where each subset is stratified by the labels array. 
    Splitting into 3 datasets is achieved by running train_test_split() twice.

    Parameters
    ----------
    dataset: the dataset to split, as a tuple (data, labels)
    frac_train, frac_val, frac_test  : The ratios with which the dataset will be split into 
             train, val, and test data. The values should be expressed as float fractions 
             and should sum to 1.0.
    shuffle: wether to suffle the datset before splitting or not.
    seed: the seed to pass to train_test_split().

    Returns
    -------
    (x_train, y_train), (x_val, y_val), (x_test, y_test) : the 3 sub-dataset.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))
    X, Y = dataset

    # note on train_test_split : Stratified train/test split is not implemented for shuffle=False 

    # Split the dataset into train and temp datasets:
    x_train, x_temp, y_train, y_temp = train_test_split(X, Y, 
                                                        stratify=Y, 
                                                        test_size=1.0-frac_train,
                                                        shuffle=True,
                                                        random_state=seed)
                                                        
    # Split the temp datafset into val and test datasets:
    relative_frac_test = frac_test / (frac_val + frac_test)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,
                                                    stratify=y_temp,
                                                    test_size=relative_frac_test,
                                                    shuffle=True,
                                                    random_state=seed)
    
    assert len(X) == len(x_train) + len(x_val) + len(x_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)



def plot_loss_accuracy(hist:list, 
                       max_epoch:float = None,
                       min_acc:float=None,  max_acc:float=None, min_loss:float=None, max_loss:float=None, 
                       training:bool=True, 
                       single_legend:bool= True,
                       figsize=(15,5),
                       message='',
                       plt_show=False):
    '''
    Plot training & validation loss & accuracy values, giving an argument
    'hist' of type 'tensorflow.python.keras.callbacks.History'. 
    '''
    
    custom_lines = [Line2D([0], [0], color='blue', lw=1, marker='o'),
                    Line2D([0], [0], color='orange', lw=1, marker='o')]
    val_colors = ('orange', 'gold', 'goldenrod', 'darkgoldenrod', 'lightcoral', 'firebrick')
        
    if not isinstance(hist, list): hist = [hist]

    if max_epoch is None:
        epoch_array = np.array(hist[0].epoch)+1
    else:
        epoch_array = np.arange(1, max_epoch+1)
    nb_epoch = len(epoch_array)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(f"Model <{message}> trained on {get_cpu_info()['brand_raw']}")

    for (i, h) in enumerate(hist):
        if h.history.get('accuracy') and training:
            ax1.plot(epoch_array, h.history['accuracy'][:nb_epoch], 'o-', markersize=4,
                     color='blue', label=f'train-{i:01d}')
        if h.history.get('val_accuracy'):
            val_color = val_colors[i % len(val_colors)]
            ax1.plot(epoch_array, h.history['val_accuracy'][:nb_epoch], 'o-', markersize=4,
                     color=val_color, label=f'val-{i:01d}')
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch') 
    y_min, y_max = ax1.get_ylim()
    if min_acc is not None: y_min = min_acc
    if max_acc is not None: y_max = max_acc
    ax1.set_ylim((y_min, y_max))
    ax1.grid(which='major', color='xkcd:cool grey',  linestyle='-',  alpha=0.7)
    ax1.grid(which='minor', color='xkcd:light grey', linestyle='--', alpha=0.5)
    if  single_legend:
        ax1.legend(custom_lines, ['Train', 'Valid'], loc='lower right')
    else:
        ax1.legend(loc='lower right')
    #ax1.set_xticks(np.arange(1, len(h.epoch)+1))
    
    
    # Plot training & validation loss values
    for (i, h) in enumerate(hist):
        if h.history.get('loss') and training:
            ax2.plot(epoch_array, h.history['loss'][:nb_epoch], 'o-', markersize=4,
                     color='blue', label=f'train-{i:01d}')
        if h.history.get('val_loss'):
            val_color = val_colors[i % len(val_colors)]
            ax2.plot(epoch_array, h.history['val_loss'][:nb_epoch], 'o-', markersize=4,
                     color=val_color, label=f'val-{i:01d}')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    y_min, y_max = ax2.get_ylim()
    if min_loss is not None: y_min = min_loss
    if max_loss is not None: y_max = max_loss
    ax2.set_ylim((y_min, y_max))
    ax2.grid(which='major', color='xkcd:cool grey',  linestyle='-',  alpha=0.7)
    ax2.grid(which='minor', color='xkcd:light grey', linestyle='--', alpha=0.5)
    if  single_legend:
        ax2.legend(custom_lines, ['Train', 'Valid'], loc='upper right')
    else:
        ax2.legend(loc='upper right')
    #ax2.set_xticks(np.arange(1, len(h.epoch)+1))

    fig.tight_layout()

    if plt_show: plt.show()
    return fig

def plot_images(image_array:np.ndarray, 
                R:int, C:int, r:int=0,
                figsize:tuple=None, 
                label_array:np.ndarray=None, 
                reverse:bool=False,
                plt_show=False):
    '''
    Plot the images from image_array on a R x C grid, starting at image rank r.
    Arguments:
       image_array: an array of images
       R: the number of rows
       C: the number of columns
       r: the starting rank in the array image_array (default: 0)
       figsize: the sise of the display (default: (C//2+1, R//2+1))
       label_array: an optional array of labels to give the imshow title
       reverse: wether to reverse video the image or not (default: False)
       titles: an optional list of titel for each imshow
    '''
    if figsize is None: figsize=(C//2+1, R//2+1 )
    fig, axes = plt.subplots(R, C, figsize=figsize)
    plt.subplots_adjust(top=.95, bottom=.01, hspace=.05, wspace=.3)
    for i, ax in enumerate(axes.flatten()):
        im = image_array[r+i]
        if reverse: im = 255 - im
        if label_array is not None: ax.set_title(str(label_array[r+i]), fontsize=8)
        ax.imshow(im, cmap='gray')
        ax.axis('off');

    if plt_show: plt.show()
    return fig

def show_conf_matrix(actual_label, predicted_labels, classes, figsize=(8,7), cmap=plt.cm.Blues, plt_show=False):

    fig  = plt.figure(figsize=figsize)
    axis = plt.axes()
    plt.title('Confusion Matrix', fontsize=16, pad=15)
    plt.ylabel('True labels', fontsize=13)
    plt.gca().xaxis.set_label_position('top') 
    plt.xlabel('Predicted labels', fontsize=13)
    plt.gca().xaxis.tick_top()
    plt.gca().figure.subplots_adjust(bottom=0.2)
    ConfusionMatrixDisplay.from_predictions(actual_label, predicted_labels, 
                                            ax=axis,
                                            cmap=cmap,
                                            display_labels=classes, 
                                            #xticks_rotation='vertical',
                                            colorbar=False);
    if plt_show: plt.show()
    return fig
    
def scan_dir(path):
    import os
    tree = ''
    data = [item for item in os.walk(path)]
    for item in data:
        if item[2]:
            for file in item[2]:
                tree += f'{item[0]}/{file}\n'
        else:
            tree += f'{item[0]}/\n'
    return tree
    
