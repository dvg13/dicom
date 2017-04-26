import numpy as np
from scipy.misc import imresize

def rescale_image(image,max_intensity=None):
    """
    rescale the image such that all of the pixel value are between 0 and 1
    :param max_intensity: maximum intensity to scale by.  Could be global max or format max, ie 255
    (Where not provided use the local max)
    """
    if max_intensity:
        image /= max_intensity
    else:
        image /= np.max(image)
    return image

def resize_image(image,height,width,mode):
    """
    resize image to specified dimensions
    :param height: height to resize to
    :param width: width to resize to
    """
    return imresize(image,[height,width],mode=mode,interp='nearest')

def mean_center(image,mean=None):
    """
    mean center the image
    param mean: global mean.  Can be size of the image to subtract pixel mean or single value
    for (1) channel mean.  (where not provided use local mean)
    """
    if mean:
        return image - mean
    else:
        return image - image.mean()
    return image

def process_image(image,size,logger=None):
    """
    process a 1-channel square image
    :param image: the image to process
    :param size: the target image size (square)
    :param logger: logger to log errors
    :return: processed image
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

def process_mask(mask,size,logger=None):
    """
    process a 1 channel, square, binary mask to 0/1 array
    :param mask: the mask to process
    :param size: the target image size (square)
    :param logger: logger to log errors
    :return: processed mask
    """
    try:
        #mask = mask.astype(np.float32)
        mask = resize_image(mask,size,size,'1').reshape(size,size,1).astype(bool)
        return mask
    except Exception:
        logger.exception("Error Processing Contour Mask")
        return None
