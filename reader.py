import numpy as np
import image_utils
import os

class Image_Mask_Reader():
    def __init__(self,files_file,batch_size,image_size,truncate_last_batch=False,
                both_contours=False,neural_net=False,):
        """
        :param files_file: str - text file containing a filename on each line
        :param batch_size: int - number of images to return per batch
        :param image_size: int - size of output image and target mask (assumes square)
        :param truncate_last_batch: bool - truncate last batch of an epoch
        :param neural_net: bool - specify whether to process the images for a neural net or
        return the raw arrays
        :param both_contours: bool - get o_contous in addition to i-contours
        """
        self.files_file = files_file
        self.batch_size = batch_size
        self.image_size = image_size
        self.truncate_last_batch = truncate_last_batch
        self.both_contours=both_contours
        self.nn = neural_net

        self.file_list = self._get_file_list(files_file)
        self.num_files = len(self.file_list)
        self.file_idx = 0
        self.epoch = 0
        self._start_epoch()

    #might want to log invalid files that get ignored
    def _get_file_list(self,filenames):
        """
        :param filenames: str - file that contains 1 filename per line
        :return: a list of these files
        """
        file_list = []
        with open(filenames) as f:
            for line in f:
                filename = line.strip()
                if os.path.exists(filename):
                    file_list.append(filename)
        return file_list

    def _start_epoch(self):
        """
        shuffle,reset the file_idx,and iterate the epoch
        """
        np.random.shuffle(self.file_list)
        self.file_idx = 0
        self.epoch += 1

    def _check_epoch(self,idx):
        if idx == self.num_files:
            self._start_epoch()

    def next(self):
        """
        :return: two or three np arrays of shape (b * size * size * 1)
        The first contains values between (-1,1)
        The second and and optional 3rd are boolean
        The first array is the image, the second is the i_contour, and the third is the o_contour
        """
        #the last batch may be smaller if it is desired to strictly run an epoch
        current_batch_size = self.batch_size if not self.truncate_last_batch \
                             else min(self.batch_size,self.num_files - self.file_idx)
        images = []
        i_contours = []
        if self.both_contours:
            o_contours = []

        for i in range(current_batch_size):

            #if not truncating the last batch - can start new epoch in the middle of a batch
            self._check_epoch(self.file_idx + i)

            current_file = self.file_list[self.file_idx + i]
            data = np.load(current_file)

            #these would raise key error
            image = data["image"]
            i_contour = data["i_contour"]
            if self.both_contours:
                o_contour = data["o_contour"]

            #processing can raise a number of errors
            images.append(image_utils.process_image(image,self.image_size) if self.nn else image)
            i_contours.append(image_utils.process_mask(i_contour,self.image_size) if self.nn else i_contour)
            if self.both_contours:
                o_contours.append(image_utils.process_mask(o_contour,self.image_size) if self.nn else o_contour)

        self.file_idx += current_batch_size
        self._check_epoch(self.file_idx)

        if self.both_contours:
            return (np.stack(images),np.stack(i_contours),np.stack(o_contours))
        else:
            return (np.stack(images),np.stack(i_contours))
