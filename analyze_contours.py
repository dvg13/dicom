import numpy as np
import matplotlib.pyplot as plt
import reader
from scipy import ndimage
from skimage import feature
from skimage.morphology import convex_hull_image
import os
from scipy.misc import imsave
from copy import copy

def scale(image,o_contour):
    """
    scale the masked portion of the image between 0 and 1
    the unmasked portion is set to 0
    """
    image = image.astype(np.float32)
    image *= o_contour
    image[np.nonzero(o_contour)] -= np.min(image[np.nonzero(o_contour)])
    image[np.nonzero(o_contour)] /= np.max(image[np.nonzero(o_contour)])

    return image

def jaccard(predicted,i_contour):
    """
    intersection over union
    """
    intersect = np.sum(predicted & i_contour)
    union = np.sum(predicted | i_contour)

    return float(intersect)/union

def remove_components(image,min_component=2000):
    """
    removes components under a certain size/intensity
    min_component: minimum size/intensity of components to keep in image
    """
    labeled_image, num = ndimage.label(image)
    for label in range(num):
        if np.sum(np.where(labeled_image == label)) < min_component:
            image[np.where(labeled_image == label)] = False
    return image

def thresh_image(image,o_contour,threshold,hull=False,min_component=None):
    """
    Threshold the input image
    threshold: can be int or float depending on scale of image
    hull: bool - convert predicted mask to convex hull
    min_component: - int - minimum size for predicted components.  If none, don't evaluate components
    """
    threshed = image * o_contour > threshold

    if min_component is not None:
        threshed = remove_components(threshed,min_component)

    if hull and np.sum(threshed):
        threshed = convex_hull_image(np.squeeze(threshed))
    return threshed

def canny_hull(image,sigma,o_contour):
    """
    get convex hull of the canny edge image
    sigma: int - parameter for guassian filter used in edge detector
    """
    image = scale(image,o_contour)
    edge = feature.canny(np.squeeze(image),sigma=sigma,mask=np.squeeze(o_contour),low_threshold=.25)
    if np.sum(edge):
        hull = convex_hull_image(edge)
        return hull
    return edge

def score_image_thresh(threshold,scale_image,hull=False,min_component=None):
    """
    get a jaccard score on a full pass of the data for a given threshold
    """
    scores = []
    image_feeder = reader.Image_Mask_Reader("training/both_contour.txt",1,256,True,True)
    while image_feeder.epoch <= 1:
        (image,i_contour,o_contour) = image_feeder.next()
        if scale_image:
            image = scale(image,o_contour)
        threshed = thresh_image(image,o_contour,threshold,hull,min_component)
        scores.append(jaccard(threshed,i_contour))
    return np.mean(scores)

def score_image_canny(sigma):
    """
    get a jaccard score on all of the data for a particular value of sigma
    """
    scores = []
    image_feeder = reader.Image_Mask_Reader("training/both_contour.txt",1,256,True,True)
    while image_feeder.epoch <= 1:
        (image,i_contour,o_contour) = image_feeder.next()
        canny_hulled = canny_hull(image,sigma,o_contour)
        scores.append(jaccard(canny_hulled,i_contour))
    return np.mean(scores)

def get_max_intensity():
    """
    get the max intensity in the dataset
    """
    max_intensity = 0
    image_feeder = reader.Image_Mask_Reader("training/both_contour.txt",1,256,True,True)
    while image_feeder.epoch <= 1:
        (image,i_contour,o_contour) = image_feeder.next()
        if np.max(image) > max_intensity:
            max_intensity = np.max(image)
    return max_intensity

def test_thresholds(filename,num=500,scale_image=True,hull=False,min_component=None):
    """
    test different threshold values on full data (here is small)
    write results to file and show a pyplot plot
    num: int - number of values to test
    scale_image: bool - scale masked image between 0 and 1
    hull: bool - convert predicted mask to convex hull
    min_component: - int - minimum size for predicted components.  If none, don't evaluate components
    """
    scores = []
    score_file = open(filename,'w')

    if not scale_image:
        max_intensity = get_max_intensity()

    for thresh in np.linspace(0,1 if scale_image else max_intensity,num):
        scores.append(score_image_thresh(thresh,scale_image,hull,min_component))
        score_file.write("{}\t{}\n".format(thresh,scores[-1]))
        print(thresh,scores[-1])
    score_file.close()
    plt.plot(np.linspace(0,1 if scale_image else max_intensity,num),scores)
    plt.show()

def test_canny(num=10):
    """
    evaluate different sigma values over the whole dataset
    num: int - number of sigma values to try
    """
    scores = []
    for sigma in range(num):
        scores.append(score_image_canny(sigma))
        print(sigma,scores[-1])

def overlay(image,mask):
    """
    create an rgb image with the input image as the blue channel
    and a mask as the green channel
    """
    return np.dstack((np.zeros_like(np.squeeze(image)),np.squeeze(image),np.squeeze(mask)))

def print_thresholded_masks(output_dir,threshold,scale_image,hull=False,min_component=None):
    """
    saves thresholded image masks
    scale_image: bool - scale masked image between 0 and 1
    threshold: int or float depending on input scaling
    hull: bool - convert predicted mask to convex hull
    min_component: - int - minimum size for predicted components.  If none, don't evaluate components
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_feeder = reader.Image_Mask_Reader("training/both_contour.txt",1,256,True,True)
    i = 0
    while image_feeder.epoch <= 1:
        (image,i_contour,o_contour) = image_feeder.next()
        temp_image = copy(image)
        if scale_image:
            temp_image = scale(temp_image,o_contour)

        threshed = thresh_image(temp_image,o_contour,threshold,hull)
        imsave(os.path.join(output_dir,str(i)+".png"),overlay(image / np.max(image),threshed))
        i+=1

def print_canny_masks(output_dir,sigma=2):
    """
    save canny-based masks
    sigma: int - parameter for guassian filter used in edge detector
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_feeder = reader.Image_Mask_Reader("training/both_contour.txt",1,256,True,True)
    i = 0
    while image_feeder.epoch <= 1:
        (image,i_contour,o_contour) = image_feeder.next()
        hull = canny_hull(copy(image),sigma,o_contour)
        imsave(os.path.join(output_dir,str(i)+".png"),overlay(image / np.max(image),hull))
        i+=1

def calculate_pixel_difference(scale_image):
    """
    calculate the difference in pixel intensities between i_contour
    and o_contour - i_contour
    scale_image: bool - scale masked image between 0 and 1
    """
    differences = []
    image_feeder = reader.Image_Mask_Reader("training/both_contour.txt",1,256,True,True)
    while image_feeder.epoch <= 1:
        (image,i_contour,o_contour) = image_feeder.next()
        if scale_image:
            image = scale(image,o_contour)

        o_pixels = o_contour & ~i_contour
        o_average = np.mean(image[o_pixels])
        i_average = np.mean(image[i_contour])

        differences.append(i_average - o_average)
    return np.mean(differences)

test_canny(10)
