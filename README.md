# Audio-Quality-Identification

##### Add the python file to the directory containing the dataset.

To get F1,Precision Recall for Silero with 32ms frames,32ms hop run the frame_wise_32_silero. It takes in 4 parameters 

* the hop size (give 32 for this file)
* threshold(a number between 0 & 1) 
* results_labels(True/False) generates new label files for each file containiing VAD annotated label files
* verbose (True/False)  to show results for individual files.

To get F1,Precision Recall for WEBRTC  run the frame_wise_WEBRTC. It takes in 4 parameters 

* hop size (10,20,30)
* agressiveness parameter(1,2,3 where 3 is most likely to be speech) 
* results_labels(True/False) generates new label files for each file containiing VAD annotated label files
* verbose (True/False)  to show results for individual files.

#### Use the iterative programs to get output arrays for various Silero thresholds values, it takes input - hop size.


#### To run the noise addition program, put the program in the same directory as the clean files and noise to be added to the "Noises" directory in the program directory, you will need to make changes to the "SNR_add" function in line 51 to intorduce new noises if any. The inputs taken by this program are:

* upper limit of desired SNR values (giving 20 would generate noise files between 1 and 20)
* number of files for each SNR value. (as each time random snippet of noise is taken multiple files of same SNR value are required)

