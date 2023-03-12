# Audio-Quality-Identification
To get F1,Precision Recall for Silero with 32ms frames,32ms hop run the frame_wise_32_silero. It takes in 4 parameters 

* the window size (give 32 for this file)
* threshold(a number between 0 & 1) 
* results_labels(True/False) generates new label files for each file containiing VAD annotated label files
* verbose (True/False)  to show results for individual files.
