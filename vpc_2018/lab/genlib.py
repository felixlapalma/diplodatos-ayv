# auxiliary funcs based on
# 0_explore-data.ipynb
# 1_data-generator.ipynb
#...
# @ https://github.com/DiploDatos/VisionPorComputadoras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools


def get_classes_distribution(y_data,LABELS):
    """
    MOD:add Labels Dict
    """
    # Get the count for each label
    y = np.bincount(y_data)
    ii = np.nonzero(y)[0]
    label_counts = zip(ii, y[ii])

    # Get total number of samples
    total_samples = len(y_data)

    # Count the number of items in each class
    for label, count in label_counts:
        class_name = LABELS[label]
        percent = (count / total_samples) * 100
        print("{:<15s}:  {} or {:.2f}%".format(class_name, count, percent))
        
    return label_counts


def plot_label_per_class(y_data,LABELS):
    """
    MOD:add Labels Dict
    """
    
    classes = sorted(np.unique(y_data))
    f, ax = plt.subplots(1,1, figsize=(12, 4))
    g = sns.countplot(y_data, order=classes)
    g.set_title("Number of labels for each class")
    
    for p, label in zip(g.patches, classes):
        g.annotate(LABELS[label], (p.get_x(), p.get_height() + 0.2))
    
    plt.show()
    
    
def sample_images_data(x_data, y_data,LABELS):
    """
    MOD:add Labels Dict
    """
    # An empty list to collect some samples
    sample_images = []
    sample_labels = []

    # Iterate over the keys of the labels dictionary defined in the above cell
    for k in LABELS.keys():
        # Get four samples for each category
        samples = np.where(y_data == k)[0][:4]
        # Append the samples to the samples list
        for s in samples:
            img = x_data[s]
            sample_images.append(img)
            sample_labels.append(y_data[s])

    print("Total number of sample images to plot: ", len(sample_images))
    return sample_images, sample_labels




def plot_sample_images(data_sample_images, data_sample_labels,LABELS, cmap="gray"):
    """
    MOD:add Labels Dict
    """
    # Plot the sample images now
    f, ax = plt.subplots(5, 8, figsize=(16, 10))

    for i, img in enumerate(data_sample_images):
        ax[i//8, i%8].imshow(img, cmap=cmap)
        ax[i//8, i%8].axis('off')
        ax[i//8, i%8].set_title(LABELS[data_sample_labels[i]])
    plt.show()    
    
def data_preprocessing_y(y_data):
    out_y = to_categorical(y_data, len(np.unique(y_data)))
    return out_y



def save_to_disk(x_data, y_data, usage, output_dir='cifar10_images'):
    """
    This function will resize your data using the specified output_size and 
    save them to output_dir.
    
    x_data : np.ndarray
        Array with images.
    
    y_data : np.ndarray
        Array with labels.
    
    usage : str
        One of ['train', 'val', 'test'].

    output_dir : str
        Path to save data.
    """
    assert usage in ['train', 'val', 'test']
    
    # Set paths 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for label in np.unique(y_data):
        label_path = os.path.join(output_dir, usage, str(label))
        if not os.path.exists(label_path):
            os.makedirs(label_path)
    
    for idx, img in enumerate(x_data):
        bgr_img = img[..., ::-1]  # RGB -> BGR
        label = y_data[idx][0]
        img_path = os.path.join(output_dir, usage, str(label), 'img_{}.jpg'.format(idx))

        retval = cv2.imwrite(img_path, bgr_img)
        assert retval, 'Problem saving image at index:{}'.format(idx)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta correcta')
    plt.xlabel('Etiqueta predicha')
