import unittest
import argparse
import sys
import reader
import numpy as np

class TestReader(unittest.TestCase):


    def test_dimensions(self):
        """
        Test the dimension of one batch
        """
        im_reader = reader.Image_Mask_Reader(args.files,16,128,False)
        result = im_reader.next()
        self.assertEqual(len(result),2)
        self.assertEqual(result[0].shape,(16,128,128,1))
        self.assertEqual(result[1].shape,(16,128,128,1))

    def test_data_range(self):
        """
        Test that the data lies within the correct ranges
        """
        im_reader = reader.Image_Mask_Reader(args.files,16,256,False)
        while im_reader.epoch <= 1:
            result = im_reader.next()

            #test image range
            self.assertTrue(np.min(result[0]) >= -1)
            self.assertTrue(np.max(result[0]) <= 1)
            self.assertTrue(abs(np.mean(result[0])) < 1e-5)

            #test masks
            self.assertEqual(result[1].dtype,bool)
            #self.assertEqual(np.max(result[1]),1)
            #self.assertEqual(np.min(result[1]),0)

    def test_epoch_size(self):
        """
        Tests the last batch truncation
        If this is set to true - should shorten last batch so as to
        process exactly 1 epoch.  If false, will keep batch size the
        same and process slightly more than 1
        """
        #truncate last batch
        im_reader = reader.Image_Mask_Reader(args.files,20,256,True)
        total = 0
        while im_reader.epoch <= 1:
            result = im_reader.next()
            total+=result[0].shape[0]
        self.assertEqual(total,96)

        #don't tuncate last batch
        im_reader = reader.Image_Mask_Reader(args.files,20,256,False)
        total = 0
        while im_reader.epoch <= 1:
            result = im_reader.next()
            total+=result[0].shape[0]
        self.assertEqual(total,100)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--files',
        type=str,
        default='training/filelist.txt'
    )
    parser.add_argument('unittest_args', nargs='*')
    args=parser.parse_args()
    sys.argv[1:] = args.unittest_args
    unittest.main()
