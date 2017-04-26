import numpy as np
import image_utils
import os

class Image_Mask_Reader():
    def __init__(self,files_file,batch_size,image_size,truncate_last_batch=False):
        """
        :param files_file: text file containing a filename on each line
        :param batch_size: number of images to return per batch
        :param image_size: size of output image and target mask (assumes square)
        :param truncate_last_batch: truncate last batch of an epoch
        """
        self.files_file = files_file
        self.batch_size = batch_size
        self.image_size = image_size
        self.truncate_last_batch = truncate_last_batch

        self.file_list = self._get_file_list(files_file)
        self.num_files = len(self.file_list)
        self.file_idx = 0
        self.epoch = 0
        self._start_epoch()

    #might want to log invalid files that get ignored
    def _get_file_list(self,filenames):
        """
        :param filenames: file that contains 1 filename per line
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

    def next(self):
        """
        :return: two np arryas of shape (b * size * size * 1)
        The first contains values between (-1,1)
        The second contains values (0 or 1)
        Both have dtype=float32
        """
        #the last batch may be smaller if it is desired to strictly run an epoch
        current_batch_size = self.batch_size if not self.truncate_last_batch \
                             else min(self.batch_size,self.num_files - self.file_idx)

        images = np.empty([current_batch_size,self.image_size,self.image_size,1],dtype=np.float32)
        targets = np.empty([current_batch_size,self.image_size,self.image_size,1],dtype=bool)

        for i in range(current_batch_size):

            #if not truncating the last batch - can start new epoch in the middle of a batch
            if self.file_idx + i == self.num_files:
                self._start_epoch()

            current_file = self.file_list[self.file_idx + i]
            data = np.load(current_file)

            images[i] = image_utils.process_image(data["image"],self.image_size)
            targets[i] = image_utils.process_mask(data["target"],self.image_size)

        self.file_idx += current_batch_size
        if self.file_idx == self.num_files:
            self._start_epoch()

        return images,targets
