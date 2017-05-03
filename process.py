import pandas as pd
import numpy as np
import os
import parsing
from scipy.misc import imsave
import logging
import argparse

class Dicom_Processor():
    def __init__(self,link_file,dicom_dir,contour_dir,output_dir):
        """
        :param link_file: csv file containing links between dicom dirs and contour dirs
        :param dicom_dir: root directory of input dicom images
        :param contour_dir: root directory of input contour images
        :param output_dir: root directory for output npz files
        """
        self.link_file = link_file
        self.dicom_dir = dicom_dir
        self.contour_dir = contour_dir
        self.output_dir = output_dir

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
        i_contour_file = open(os.path.join(self.output_dir,"i_contour.txt"),'w')
        o_contour_file = open(os.path.join(self.output_dir,"o_contour.txt"),'w')
        both_contour_file = open(os.path.join(self.output_dir,"both_contour.txt"),'w')

        links_df = pd.read_csv(self.link_file)

        for row in links_df.itertuples():

            #get directory names and test that they exist
            patient_dicom_dir = os.path.join(self.dicom_dir,row.patient_id)
            patient_i_contour_dir = os.path.join(self.contour_dir,row.original_id,'i-contours')
            patient_o_contour_dir = os.path.join(self.contour_dir,row.original_id,'o-contours')

            #test that the dicom directory exists
            if not self._test_dir(patient_dicom_dir):
                break

            #get files and image numbers
            dicom_id_file_dict = self._dir_to_id_file_dict(patient_dicom_dir,self._dicom_id_parser)
            i_contour_id_file_dict = self._dir_to_id_file_dict(patient_i_contour_dir,self._contour_id_parser)
            o_contour_id_file_dict = self._dir_to_id_file_dict(patient_o_contour_dir,self._contour_id_parser)

            for image_id in dicom_id_file_dict:
                have_i_contour = image_id in i_contour_id_file_dict
                have_o_contour = image_id in o_contour_id_file_dict

                #process if either mask exists
                if have_i_contour or have_o_contour:
                    npz_items = {}

                    #parse the dicom image
                    dicom_dict = parsing.parse_dicom_file(dicom_id_file_dict[image_id],self.logger)
                    if dicom_dict is not None:
                        dicom_image = dicom_dict['pixel_data']
                        npz_items["image"] = dicom_image
                    else:
                        break

                    #parse the i_contour
                    if have_i_contour:
                        i_contour_mask = self._get_contour_mask(i_contour_id_file_dict[image_id],dicom_image.shape)
                        if i_contour_mask is not None:
                            npz_items["i_contour"] = i_contour_mask
                        else:
                            have_i_contour = False

                    #parse the o_contour
                    if have_o_contour:
                        o_contour_mask = self._get_contour_mask(o_contour_id_file_dict[image_id],dicom_image.shape)
                        if o_contour_mask is not None:
                            npz_items["o_contour"] = o_contour_mask
                        else:
                            have_o_contour=False

                    #save the npz
                    if have_i_contour or have_o_contour:
                        #create the directory if it doesn't exist
                        patient_dir = os.path.join(self.output_dir,row.patient_id)
                        self._create_dir(patient_dir)

                        #save
                        npz_path = os.path.join(patient_dir,str(image_id))
                        np.savez(npz_path,**npz_items)

                        #write the npz filename to the appropriate lists
                        if have_i_contour:
                            i_contour_file.write(npz_path + ".npz\n")
                        if have_o_contour:
                            o_contour_file.write(npz_path + ".npz\n")
                        if have_i_contour and have_o_contour:
                            both_contour_file.write(npz_path + ".npz" + "\n")

        i_contour_file.close()
        o_contour_file.close()
        both_contour_file.close()

def main(args):
    processor=Dicom_Processor(args.link,args.dicom_dir,args.contour_dir,args.output_dir)
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
    args = parser.parse_args()
    main(args)
