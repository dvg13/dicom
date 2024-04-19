# dicom
### Part 1:
The process.py script performs the task described in part one.  The usage is:
```
python process.py 
--link=<path to links csv> 
--dicom_dir=<path to dicom root dir>
--contour_dir=<path to contour files root dir> 
--outut_dir=<root dir for output files>
--print_overlaid=<True/False> - print the images overlaid with the masks (default False)
(all variables default to the locations in the final_data directory)
```

Notes:
1) I chose to save the data to .npz files.  Depending on the application, it might be
useful to store them as images so that can be inspected more easily.  I chose to use .npz files as they 
naturally pair the images and masks, and don't modify the raw values of the images in case
we want to process them in different ways later.
2) As it was small, I pushed the results of running code to the repo.  It creates the .npz files in directories
named with the patientid from the links file.  It also creates filelist.txt, which contains the 
paths to all of the succesfully created .npz files. 
3) It also outputs a log file.  This will log error messages for all files that raised an exception.  There weren't any of these in the data provided, so this is blank.  
4) I verified the correctness of the data in two ways.  Firstly, I overlaid the masks on top of dicom images,
so that I could see that they line up - which they more or less do.  This functionality is provided by the --print-overlaid
flag.  It prints all of the overlaod images to the training/overlaid directory.  I also verified the correctness in the unit
tests for the reader in part 2.  This tests for the data ranges of the images - so I ensure that these are reasonable.  
5) As the code seemed to work, I didn't significantly modify the given file.  I added some exceptions and logging to the contour file reader, so that if the contour files are not in the right format it will exit graceful and log the error. 

### Part 2:
The reader.py script performs the task described in part two.  It doesn't take command line arguments as it's presumably getting called from inside the deep learning framework, but the class takes the following parameters:
```
files_file : filelist.txt created in part 1
batch_size
image_size
truncate_last_batch(False) : determines whether to truncate the last batch so as to run exactly one epoch where
the batch size does not divide evenly into the number of examples
```
I provided unit tests in the scipt reader_tests.py

Notes:
1) I didn't modify the code from part 1.  One thing that might be useful would be to do the image processing in 
Part 1 instead of in the reader - so as not to have to do it multiple times.  In this version, it's very light image processing, but it would be more efficient.  In general, using multi-threaded code to do the image loading would be a plus.
2) I created several unit tests.  They test that the batch size is correct, that it correctly stops after processing an epoch,
and that the images produced have values that lie in reasonable ranges based on the pre-processing.  Given more time, I would have had more tests for all of the code.  
3) If given more time I would have provided a version of the reader that loads the images into memory - which would be much faster on small datasets.  This also doesn't provide a train/validation split.  Because the files are read from a list - this would only involve creating a training files list and a validation files list - but one would have to be careful to keep the patients from being in both groups.  The code would have to be reworked to provide for image augmentation, which would be a nice feature.  Lastly, as mentioned above, using multi-threaded code would speed it up when I am loading the images from files.  


