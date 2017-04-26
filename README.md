# dicom

As a note I wrote and tested this with python3.

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
1) I chose to output the np arrays to npz files.  Depending on the application, it might be
useful to store them as images so that can be inspected.  I chose to use .npz files as they 
naturally pair the images and masks, and don't modify the raw values of the images in case
we want to process them in different ways.
2) As it was small, I pushed the results of running code to the repo.  It creates the npz files in directories
named with the patients name.  It also outputs a filelist to filelist.txt, which contains the 
paths to all of the succesfully created npz files. 
3) It also outputs a log file.  The code logs files that couldn't be created for various reasons, but there
aren't any in the data provided.  
4) I verified the correctness of the data in two ways.  Firstly, I overlaid the masks on top of dicom images,
so that I could see that they line up - which they more or less do.  This functionality is provided by the --print-overlaid
flag.  It prints all of the overlaod images to the training/overlaid directory.  I also verified the correctness in the unit
tests for the reader in part2.  This tests for the data ranges of the images - so I ensure that these are reasonable.  
5) As the code seemed to work, I didn't significantly modify it.  I added some exceptions and logging to the contour map function,
so that if the contour files were not in the right format it will exit graceful and log it.  

### Part 2:
The reader.py script performs the task described in part two.  It doesn't take command arguments as presumably this is getting
called from inside the deep learning framework, but the class takes the following parameters:

