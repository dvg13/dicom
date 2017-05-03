import numpy as np
from scipy.misc import imresize

def rescale_image(image,max_intensity=None):
    """
    rescale the image such that all of the pixel values are between 0 and 1
    :param max_intensity: int or floa - maximum intensity to scale by.
    Could be global max or format max, ie 255.  If None, user the local max
    """
    if max_intensity:
        image /= max_intensity
    else:
        image /= np.max(image)
    return image

def resize_image(image,height,width,mode):
    """
    resize image to specified dimensions
    :param height: int - height to resize to
    :param width: int - width to resize to
    """
    return imresize(image,[height,width],mode=mode,interp='nearest')

def mean_center(image,mean=None):
    """
    mean center the image
    :param mean: numeric or np.array of dtype numberic - The global mean.
    Can be size of the image to give the pixel mean or a single value for
    for (1) channel mean.  Where not provided use local mean.
    """
    if mean:
        return image - mean
    else:
        return image - image.mean()
    return image

def process_image_nn(image,size,logger=None):
    """
    process a 1-channel square image for use in a neural net
    returns float image of size [size,size,1]
    that was rescaled between 0 and 1 and then mean centered
    :param image: np array - the image to process
    :param size: int - the target image size (square)
    :param logger: logger to log errors
    :return: bool np array of size [size,size,1]
    """
    try:
        image = image.astype(np.float32)
        image = resize_image(image,size,size,'F').reshape(size,size,1)
        image = rescale_image(image)
        image = mean_center(image)
        return image
    except Exception:
        logger.exception("Error Processing Image")
        return None

def process_mask_nn(mask,size,logger=None):
    """
    process a 1 channel, square, binary mask for neural net
    :param mask: np boolean array - the mask to process
    :param size: int - the target image size (square)
    :param logger: logger to log errors
    :return: np boolean array of size [size,size,1]
    """
    try:
        mask = resize_image(mask,size,size,'1').reshape(size,size,1).astype(bool)
        return mask
    except Exception:
        logger.exception("Error Processing Contour Mask")
        return None
