import pandas as pd
import numpy as np
import os
import parsing
from scipy.misc import imsave
import logging
import argparse

class Dicom_Processor():
    def __init__(self,link_file,dicom_dir,contour_dir,output_dir,print_overlaid=True):
        """
        :param link_file: csv file containing links between dicom dirs and contour dirs
        :param dicom_dir: root directory of dicom images
        :param contour_dir: root directory of contour images
        :param output_dir: root directory for output npz files
        :param print_overlaid:  Flag for printing images with the mask overlaid on the image
        """
        self.link_file = link_file
        self.dicom_dir = dicom_dir
        self.contour_dir = contour_dir
        self.output_dir = output_dir
        self.print_overlaid = print_overlaid

        #initiaize logger
        self._create_dir(self.output_dir)
        self.logger = logging.getLogger('Dicom_Processor')
        log_path = os.path.join(self.output_dir,'Dicom_Processor.log')
        self.logger.addHandler(logging.FileHandler(log_path))

    def _dicom_id_parser(self,filename):
        """
        :param filename: filename to dicom image
        :return: int value corresponding to filename
        """
        return int(filename.split(".")[0])

    def _contour_id_parser(self,filename):
        """
        :param filename: filename to dicom image
        :return: int value corresponding to filename
        """
        return int(filename[8:12])

    def _dir_to_id_file_dict(self,directory,parser):
        """
        :param directory: directory containing files
        :param parser: function to extract int id from a filename
        :return:dict of <int it><filename> pairs
        """
        id_file_dict = {}

        for dirs,subsirs,files in os.walk(directory):
            for f in files:
                if f.endswith("dcm") or f.endswith("txt"):
                    try:
                        id_file_dict[parser(f)] = os.path.join(directory,f)
                    except ValueError:
                        logging.error("{} has a filename that cannot be parsed".format(os.path.join(directory,f)))
        return id_file_dict

    def _get_contour_mask(self,filename,image_size):
        """
        :param filename: contour filename
        :param image_size: size of corresponding dicom image
        :return: np boolean array representing contour
        """
        contours = parsing.parse_contour_file(filename,self.logger)
        if contours is None:
            return None
        return parsing.poly_to_mask(contours, image_size[1], image_size[0])

    def _overlay(self,image,mask):
        """
        :param image: dicom image as np array
        :param mask: mask as np array
        :return: 3-channe np array where image is blue channel and mask is green
        """
        r = np.zeros_like(image)
        g = mask * 255
        b = image
        return np.dstack((r,g,b))

    def _print_overlaid_image(self,image,mask,patient,num):
        """
        :param image: dicom image as np array
        :param mask: mask as np array
        :patient: str id
        :num: int number of the image
        """
        overlaid = self._overlay(image,mask)
        fname = os.path.join(self.output_dir,"overlaid",patient + str(num) + ".png")
        imsave(fname,overlaid)

    def _create_dir(self,dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def _test_dir(self,dir_path,log_error=True):
        """
        test whether dir_path exists and log error if not
        """
        if not os.path.exists(dir_path):
            if log_error:
                self.logger.error("{} directory does not exist".format(dir_path))
            return False
        return True

    def process(self):
        """
        Create npz files containing input/i-contour/o-contour arrays
        Creates directory sturction <output_dir>/<patient_id>/<image_num>.npz
        """
        if self.print_overlaid:
            self._create_dir(os.path.join(self.output_dir,"overlaid"))

        i_contour_file = open(os.path.join(self.output_dir,"i_contour.txt"),'w')
        both_contour_file = open(os.path.join(self.output_dir,"both_contour.txt"),'w')

        links_df = pd.read_csv(self.link_file)

        for row in links_df.itertuples():

            #get directory names and test that they exist
            patient_dicom_dir = os.path.join(self.dicom_dir,row.patient_id)
            patient_i_contour_dir = os.path.join(self.contour_dir,row.original_id,'i-contours')
            patient_o_contour_dir = os.path.join(self.contour_dir,row.original_id,'o-contours')

            #assuming o_contour_dir works
            if not (self._test_dir(patient_dicom_dir) and self._test_dir(patient_i_contour_dir)):
                break

            #get files and image numbers
            dicom_id_file_dict = self._dir_to_id_file_dict(patient_dicom_dir,self._dicom_id_parser)
            i_contour_id_file_dict = self._dir_to_id_file_dict(patient_i_contour_dir,self._contour_id_parser)
            o_contour_id_file_dict = self._dir_to_id_file_dict(patient_o_contour_dir,self._contour_id_parser)

            #create npz files for a given patient containing image and target
            patient_dir = os.path.join(self.output_dir,row.patient_id)
            self._create_dir(patient_dir)
            for image_id in dicom_id_file_dict:
                if image_id in i_contour_id_file_dict:

                    dicom_image = parsing.parse_dicom_file(dicom_id_file_dict[image_id],
                                                           self.logger)['pixel_data']
                    i_contour_mask = self._get_contour_mask(i_contour_id_file_dict[image_id],dicom_image.shape)

                    if dicom_image is not None and i_contour_mask is not None:
                        npz_path = os.path.join(patient_dir,str(image_id))
                        np.savez(npz_path+"_i_contour",image=dicom_image,i_contour=i_contour_mask)
                        i_contour_file.write(npz_path + "_i_contour.npz" + "\n")

                        #create overlay images
                        if self.print_overlaid:
                            self._print_overlaid_image(dicom_image,contour_mask,row.patient_id,image_id)

                        #create a file with both contours
                        if image_id in o_contour_id_file_dict:
                            o_contour_mask = self._get_contour_mask(o_contour_id_file_dict[image_id],dicom_image.shape)
                            np.savez(npz_path+"_both_contour",image=dicom_image,i_contour=i_contour_mask,
                                     o_contour=o_contour_mask)
                            both_contour_file.write(npz_path + "_both_contour.npz" + "\n")

        i_contour_file.close()
        both_contour_file.close()

def main(args):
    processor=Dicom_Processor(args.link,args.dicom_dir,args.contour_dir,args.output_dir,
                              args.print_overlaid)
    processor.process()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--link',
        type=str,
        default='final_data/link.csv',
        help='CSV file containing the links between dicom dirs and contour dirs'
    )
    parser.add_argument(
        '--dicom_dir',
        type=str,
        default='final_data/dicoms',
        help='root directory to dicom files'
    )
    parser.add_argument(
        '--contour_dir',
        type=str,
        default='final_data/contourfiles',
        help='root directory to contour files'
    )
    parser.add_argument(
         '--output_dir',
         type=str,
         default='training',
         help='root output directory'
    )
    parser.add_argument(
        '--print_overlaid',
        action='store_true',
        help='print overlayed images to allow for visual check'
    )
    args = parser.parse_args()
    main(args)
