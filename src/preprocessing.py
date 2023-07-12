import numpy as np
from random import randint, uniform, choice
from scipy.ndimage import shift as scipy_shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import rotate

from plot_functions import show_data_augmentation


def apply_color_borders(img):
    """Converts all pixels with values greater than 255 to 255 and all pixels
    with values less than 0 to 0."""
    img = np.maximum(img, np.zeros((28, 28)))
    img = np.minimum(img, np.ones((28, 28)) * 255)
    return img


def chack_font_thickness(data, labels):
    """Calculate number of white pixels on image for every digit. This helps detect
    whether a number is written in a narrower or wider font than average."""
    avg_thickness = []
    for i in range(10):
        images_i = data[labels == i]
        avg_thickness.append(
            np.sum(images_i) / (len(images_i) * 255)
        )
    return avg_thickness


def change_font(img, label, avg_thickness):
    """Change font of digit on image. Depending on the width of the digit of the image,
    the function expands or narrows the font. The font is wider if the font is thinner
    than usual, and thinner if it is wider than usual."""
    shifted_img = np.roll(img, randint(-1, 1), axis=0)
    shifted_img = np.roll(shifted_img, randint(-1, 1), axis=1)

    if np.sum(img) / 255 < avg_thickness[label] - 15:
        new_img = np.maximum(img, shifted_img)
    elif np.sum(img) / 255 >= avg_thickness[label] + 15:
        new_img = np.minimum(img, shifted_img)
    else:
        new_img = shifted_img
    return new_img


def shift_image(img):
    """Move (shift) image left or right, up or down for maximum 2 pixels."""
    new_img = scipy_shift(img, (uniform(-2, 2), uniform(-2, 2)), order=4)
    new_img = apply_color_borders(new_img)
    return new_img


def rotate_image(img):
    """Rotate image by the angle between -10 and 10 """
    new_img = rotate(img, angle=choice((-10, 10)), reshape=False)
    new_img = apply_color_borders(new_img)
    return new_img


def elastic_transform(image, alpha_range, sigma, random_state=None):
    """This function was copied from github and a simple image sharpening filter was
       added at the end. Below is the original description:
       Elastic deformation of images as described in [Simard2003]_.
       [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

   Arguments
       image: Numpy array with shape (height, width, channels).
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields."""

    if random_state is None:
        random_state = np.random.RandomState(None)

    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')#, np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))#, np.reshape(z, (-1, 1))

    new_img_2d = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    new_img_2d = apply_color_borders(new_img_2d)
    # Counter gaussian effect
    new_img_2d = np.multiply(new_img_2d - 100, 1.25) + 100
    return new_img_2d


def data_augmentation(data, labels, plots=False):
    """Generate new images for the dataset created by modifying existing images. Different methods of data
    augmentation and their hyperparameters are tested and those with the best results are selected."""

    #avg_thickness = chack_font_thickness(data, labels)  # for change_font()

    new_data, new_labels = [], []
    for i, img in enumerate(data):
        img_2d = np.reshape(img, (-1, 28))

        # Perform transformations:
        new_img_2d = elastic_transform(img_2d, 7.5, 1.5)
        #new_img_2d = change_font(new_img_2d, labels[i], avg_thickness)
        new_img_2d = shift_image(new_img_2d)
        new_img_2d = rotate_image(new_img_2d)

        if plots and i < 5:
            show_data_augmentation(img_2d, new_img_2d)

        new_data.append(new_img_2d.flatten())
        new_labels.append(labels[i])

    return np.concatenate((np.array(new_data), data), axis=0), np.concatenate((new_labels, labels), axis=0)
