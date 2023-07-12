import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def show_sample(img_1d, title=''):
    """Shows one image from dataset"""
    img_2d = np.reshape(img_1d, (-1, 28))
    plt.title(title)
    plt.imshow(img_2d)
    plt.show()


def print_label_counters(y_train):
    """Prints counters of the same labels (images of the same numbers)"""
    counts, _ = np.histogram(y_train)
    for i in range(10):
        print('label "{}", counts: {}'.format(i, counts[i]))
    print()


def show_data_augmentation(img_before, img_after):
    """Shows one original image from dataset paired with the image
    obtained when data augmentation is applied to original image"""
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img_before)
    axarr[0].title.set_text('Original image')
    axarr[1].imshow(img_after)
    axarr[1].title.set_text('Image after data augmentation')
    plt.show()


def plot_history(history):
    """Plot model loss and accuracy history through epochs"""
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history['val_loss'], label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)
    ax[1].plot(history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()


def show_confusion_matrix(y_pred, y_real):
    """Shows the confusion matrix (matching predicted labels with real labels)"""
    confusion_matrix = np.zeros((10, 10))
    for i, _ in enumerate(y_pred):
        confusion_matrix[np.argmax(y_pred[i]), y_real[i]] += 1
    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in range(10)], columns=[i for i in range(10)])

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.show()
