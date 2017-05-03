import argparse
import os
import numpy as np
from scipy.misc import imsave


def overlay_both(image,i_contour,o_contour):
    #need to scale image so the intensities align with masks
    r = i_contour
    g = o_contour
    b = image.astype(np.float32) / np.max(image)
    return np.dstack((r,g,b))

def overlay_one(image,contour):
    #need to scale image so the intensities align with masks
    r = np.zeros_like(contour)
    g = contour
    b = image.astype(np.float32) / np.max(image)
    return np.dstack((r,g,b))

def get_filename(output_dir,fname):
    fname_parts = fname.split("/")
    fname_file = "_".join(fname_parts[-2:])[:-4] + "png"
    return os.path.join(output_dir,fname_file)

def print_overlaid(output_dir,filelist,contours):
    with open(filelist) as files:
        for f in files:
            #try:
            if 1 == 1:
                #load data
                data = np.load(f.strip())

                image = data["image"]

                i_contour = None
                if "i_contour" in data:
                    i_contour = data["i_contour"]

                o_contour = None
                if "o_contour" in data:
                    o_contour = data["o_contour"]

                #overlay both images
                if len(contours) == 2 and i_contour is not None and o_contour is not None:
                    imsave(get_filename(output_dir,f),overlay_both(image,i_contour,o_contour))

                elif "I" in contours and i_contour is not None:
                    overlaid = overlay_one(image,i_contour)
                    imsave(get_filename(output_dir,f),overlay_one(image,i_contour))

                elif "O" in contours and o_contour is not None:
                    imsave(get_filename(output_dir,f),overlay_one(image,o_contour))

            #Add logging to this in the future
            #except:
            #    print("An error occured while processing file {}".format(f))

def main(args):
    if args.contours is not None:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        print_overlaid(args.output_dir,args.filelist,args.contours)
    else:
        print("Either --i_contours or --o_contours must be specified")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filelist',
        type=str,
        default='training/i_contour.txt',
        help='files to overlay'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='training/overlaid',
        help='directory to save overlaid images into'
    )
    parser.add_argument(
        '--i_contour',
        dest='contours',
        action='append_const',
        const="I",
        help='add the i contour to the overlaid file'
    )
    parser.add_argument(
        '--o_contour',
        dest='contours',
        action='append_const',
        const="O",
        help='add the o contour to the overlaid file'
    )
    args = parser.parse_args()
    main(args)
