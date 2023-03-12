from glob import glob
import sys
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.io


wavfiles=glob("*.wav")

SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint

USE_ONNX = False # change this to True if you want to test onnx model
if USE_ONNX:
    os.system("pip install -q onnxruntime")
  
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

verbose=sys.argv[4]

def probs(aud):
    list1=[]

    # USE_ONNX = False # change this to True if you want to test onnx model
    # if USE_ONNX:
    #     os.system("pip install -q onnxruntime")
      
    # model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
    #                               model='silero_vad',
    #                               force_reload=True,
    #                               onnx=USE_ONNX)

    # (get_speech_timestamps,
    #  save_audio,
    #  read_audio,
    #  VADIterator,
    #  collect_chunks) = utils

    ## just probabilities

    vad_iterator = VADIterator(model)
    wav = read_audio(aud, sampling_rate=16000)
    speech_probs = []
    window_size_samples = 512
    for i in range(0, len(wav),512):
        chunk = wav[i: i+ window_size_samples]
        if len(chunk) < window_size_samples:
          break
        speech_prob = model(chunk, SAMPLING_RATE).item()
        speech_probs.append([speech_prob,"start: "+str((i/16000)),"end: "+str((i+window_size_samples)/16000)])
    vad_iterator.reset_states() # reset model states after each audio

    #print(speech_probs[:,0]) # first 10 chunks predicts

    result_array=np.zeros(len(speech_probs))
    for i in enumerate(speech_probs):
        result_array[i[0]]=i[1][0]
        
    new1=(result_array>float(sys.argv[2]))
    A=np.zeros(len(new1))
    new1=list(new1)
    A[new1]=1
    #print("A = ",A)

    return A

def plotter(audio,frame_level,vad_output):
    sr,data=scipy.io.wavfile.read(audio, mmap=False)
    length=data.shape[0]/sr
    time=np.arange(len(data))*(1/16000)
    time2=[]

    for i,x in enumerate(frame_level):
        time2.append(i/16000)
    time2=np.array(time2)*512
    print("TIME",time2)
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.plot(time, data, color = color)

    ax2 = ax1.twinx()
     
    print("VAD OUT = ",vad_output)
    color = 'tab:red'
    #ax2.set_ylabel('Y2-axis', color = 'black')
    ax2.plot(time2,frame_level, color = color)
    ax2.plot(time2,0.5*vad_output, color = "purple")
    ax2.tick_params(axis ='y', labelcolor = "black")
    plt.savefig(audio[:-4]+"_plot_32ms_0.16.png")

    plt.show()    


def cont_frames(frame_arr,threshold=0.75):
    #print("THESE ARE THE FRAMES",frame_arr)
    result_array1=np.zeros(len(frame_arr))
    list1=[]
    trigger=False
    ring_buff=np.zeros(10)
    #ring_buff=deque(maxlen=10)
    per1=percentage_high_frames(frame_arr[:10],threshold)
    if(per1>0.9):
        trigger=True
        
    for i in range(len(frame_arr)):
        temp_arr=np.zeros(10)
        temp_arr2=frame_arr[i:i+10]
        temp_arr[0:len(temp_arr2)]=temp_arr2
        ring_buff=temp_arr
       # print(ring_buff)
        per=percentage_high_frames(ring_buff,threshold)
       # print("PER is",per)
        if(trigger):
            #print("entered 1")
            list1.append(1)
            result_array1[i]=1
        if (trigger==False and (per>=0.7)):
            #print("entered 2")
            trigger=True;
            result_array1[i]=1
        if (trigger==True and per<=0.1):
            #print("entered 3")
            result_array1[i:i+10]=1
            trigger=False;
    #print(result_array1)
    return result_array1
        
    
def percentage_high_frames(in_arr,threshold):
    count=0
    tot_len=len(in_arr)
    for i in in_arr:
        if i>threshold:
            count+=1
    #print(count/tot_len)
    return (count/tot_len)
    
def frames_from_labels(label_list_1,len_frame_level,frame_len,frame_level,audio,result_labels=sys.argv[3],verbose=sys.argv[4]):
    positives=np.zeros(len_frame_level)
    cleans=np.zeros(len_frame_level)
    cleans_backs=np.zeros(len_frame_level)
    nsb=np.zeros(len_frame_level)
    csnsb=np.zeros(len(frame_level))
    mb=np.zeros(len(frame_level))

    if (result_labels=="True"):
        file1= open(audio[:-4]+"_frame_level_32.txt","w")
    
    

    #print(len_frame_level)
    times=np.arange((len_frame_level))*(frame_len/1000)
    #print(times)
    for i in range(len(label_list_1)):
        current_label=label_list_1[i][2]
        if(i!=0):
            previous_label=label_list_1[i-1][2]
        else:
            previous_label=current_label
        label_start=label_list_1[i][0]
        label_end=label_list_1[i][1]
        #print("CURRENT LABEL",current_label)
        
        if(current_label=="NSB"):
            if(previous_label=="CSNSB" or previous_label=="CS" or previous_label=="CSSB"):
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)+1

                if(verbose=="True"): 
                    print("NSB Start",times[frame_start])
                nsb_start=str(times[frame_start])

                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)

                if frame_end>=len_frame_level:
                    if(verbose=="True"): 
                        print("NSB end",times[-1]+0.032)
                    nsb[frame_start:]=1
                    nsb_end=str(times[-1]+0.032)
                else:
                    if(verbose=="True"): 
                        print("NSB End",times[frame_end+1])
                    nsb_end=str(times[frame_end+1])
                nsb[frame_start:frame_end]=1
                if (result_labels=="True"):
                    file1.write((nsb_start)+"\t\t"+(nsb_end)+"\t\t"+current_label+"\n")
            else:   
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)+1
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)
                if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
                if (frame_start<len_frame_level):
                    if(verbose=="True"): 
                        print("NSB Start",times[frame_start])
                    nsb_start=str(times[frame_start])
                else:
                    if(verbose=="True"): 
                        print("NSB Start",times[frame_start-1])
                    nsb_start=str(times[frame_start-1])
                if(verbose=="True"): 
                    print("NSB End",times[frame_end])
                nsb_end=str(times[frame_end])
                nsb[frame_start:frame_end]=1
                if (result_labels=="True"):
                    file1.write((nsb_start)+"\t\t"+(nsb_end)+"\t\t"+current_label+"\n")
        if(current_label=="MB"):
            if(previous_label=="CSNSB" or previous_label=="CS" or previous_label=="CSSB"):
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)+1
                if(verbose=="True"): 
                    print(times[frame_start])
                mb_start=str(times[frame_start])

                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)
                if frame_end>=len_frame_level:
                    if(verbose=="True"): 
                        print(times[-1]+0.032)
                    mb[frame_start:]=1
                    mb_end=str(times[-1]+0.032)
                else:
                    if(verbose=="True"): 
                        print(times[frame_end+1])
                    mb_end=str(times[frame_end+1])
                mb[frame_start:frame_end]=1
                if (result_labels=="True"):
                    file1.write((mb_start)+"\t\t"+(mb_end)+"\t\t"+current_label+"\n")
            else:   
                #print("MB END",label_end)
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)
                if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
                if(verbose=="True"): 
                    print("MB Start",times[frame_start])
                mb_start=str(times[frame_start])
                if(verbose=="True"): 
                    print("MB End",times[frame_end])
                mb_end=str(times[frame_end])
                mb[frame_start:frame_end]=1
                if (result_labels=="True"):
                    file1.write((mb_start)+"\t\t"+(mb_end)+"\t\t"+current_label+"\n")
        if(current_label=="CSNSB"):  
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)+1
                if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
                if(verbose=="True"): 
                    print("CSNB Start",times[frame_start])
                csnsb_start=str(times[frame_start])
                if(verbose=="True"): 
                    print("CSNB END",times[frame_end])
                csnsb_end=str(times[frame_end])
                csnsb[frame_start:frame_end+1]=1
                if (result_labels=="True"):
                    file1.write((csnsb_start)+"\t\t"+(csnsb_end)+"\t\t"+current_label+"\n")

        if(current_label=="CS" or current_label=="CSSB" or current_label=="CSNSB"):
            frame_start=float(label_start)/(frame_len/1000)
            frame_start=int(frame_start)
            frame_end=float(label_end)/(frame_len/1000)
            frame_end=int(frame_end)+1
            if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
            positives[frame_start:frame_end+1]=1
            #print("Positives_array",positives)
            #print("FRAME START END",frame_start*(frame_len/1000),frame_end*(frame_len/1000))


        if(current_label=="CS"):  
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)+1
                if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
                if(verbose=="True"): 
                    print("CS Start",times[frame_start])
                cs_start=str(times[frame_start])
                if(verbose=="True"): 
                    print("CS END",times[frame_end])
                cs_end=str(times[frame_end])
                cleans[frame_start:frame_end+1]=1
                if (result_labels=="True"):
                    file1.write((cs_start)+"\t\t"+(cs_end)+"\t\t"+current_label+"\n")

        if(current_label=="CSSB"):  
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)+1
                if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
                if(verbose=="True"): 
                    print("CSSB Start",times[frame_start])
                cssb_start=str(times[frame_start])
                if(verbose=="True"): 
                    print("CSSB END",times[frame_end])
                cssb_end=str(times[frame_end])
                cleans_backs[frame_start:frame_end+1]=1
                if (result_labels=="True"):
                    file1.write((cssb_start)+"\t\t"+(cssb_end)+"\t\t"+current_label+"\n")

        if(current_label=="SIL"):
            if(previous_label=="CSNSB" or previous_label=="CS" or previous_label=="CSSB"):
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)+1
                if(frame_start<len_frame_level):
                    if(verbose=="True"): 
                        print("SIL Start",times[frame_start])
                    sil_start=str(times[frame_start])
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)

                if frame_end>=len_frame_level:
                    if(verbose=="True"): 
                        print("SIL end",times[-1]+0.032)
                    nsb[frame_start:]=1
                    sil_end=str(times[-1]+0.032)
                else:
                    if(verbose=="True"): 
                        print("SIL End",times[frame_end])
                    sil_end=str(times[frame_end])
                nsb[frame_start:frame_end]=1
                if (result_labels=="True"):
                    file1.write((sil_start)+"\t\t"+(sil_end)+"\t\t"+current_label+"\n")
            else:   
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)+1
                if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
                if(verbose=="True"): 
                    print("SIL Start",times[frame_start])
                sil_start=str(times[frame_start])
                if(verbose=="True"): 
                    print("SIL End",times[frame_end])
                sil_end=str(times[frame_end])
                nsb[frame_start:frame_end]=1
                if (result_labels=="True"):
                    file1.write((sil_start)+"\t\t"+(sil_end)+"\t\t"+current_label+"\n")

        # if(current_label=="CSSB"):
        #     frame_start=float(label_start)/(frame_len/1000)
        #     frame_start=int(frame_start)
        #     frame_end=float(label_end)/(frame_len/1000)
        #     frame_end=int(frame_end)
        #     cleans_backs[frame_start:frame_end+1]=1

        #     print("CSSB Start",times[frame_start])
        #     print("CSSB End",times[frame_end])
        #     #print("Positives_array",positives)
        #     #print("FRAME START END",frame_start*(frame_len/1000),frame_end*(frame_len/1000))
        #     print("Clean Backs start",cleans_backs[frame_start-10:frame_start+1])
        #     print("Clean Backs end",cleans_backs[frame_end:frame_end+10])


    if (result_labels=="True"):
        file1.close()
    if(verbose=="True"): 
        print("pos, cssb,csnsb,cs",get_num_ones(positives),get_num_ones(cleans_backs),get_num_ones(csnsb),get_num_ones(cleans))
        print("Frame level labeling",positives)
        print("Frame level Vad Output",frame_level)
    return positives,cleans,cleans_backs,nsb,csnsb,mb;


def get_num_ones(arr):
    nums=0
    for i in arr:
        if i==1:
            nums+=1
    return nums

def get_accuracy(output_array,voiced_indexes,not_speech_indexes,cs_indexes,cssb_indexes,nsb_indexes,csnsb_indexes,mb_indexes):
    true_positives=0
    cs=0
    cssb=0
    cssb_accuracy=0
    nsb=0
    overall_accuracy=0
    nsb_accuracy=0
    csnsb=0
    mb=0
    false_positives=0
    false_negatives=0
    #cs_accuracy=cs/len(cs_indexes)
    for i in cs_indexes:
        if(output_array[i]==1):
            cs+=1
    for i in cssb_indexes:
        if(output_array[i]==1):
            cssb+=1
    for i in voiced_indexes:
        if(output_array[i]==1):
            true_positives+=1
        else:
            false_negatives+=1
    for i in nsb_indexes:
        if(output_array[i]==1):
            nsb+=1
    for i in mb_indexes:
        if(output_array[i]==1):
            mb+=1
    for i in csnsb_indexes:
        if(output_array[i]==1):
            csnsb+=1
    for i in not_speech_indexes:
        if(output_array[i]==1):
            false_positives+=1
    # if(len(voiced_indexes)!=0):
    #     overall_accuracy=true_positives/len(voiced_indexes)
    # if(len(cs_indexes)!=0):
    #     cs_accuracy=cs/len(cs_indexes)
    # if(len(cssb_indexes)!=0):
    #     cssb_accuracy=cssb/len(cssb_indexes)
    # if(len(nsb_indexes)!=0):
    #     nsb_accuracy=nsb/len(nsb_indexes)

    overall_accuracy=np.array((true_positives,len(voiced_indexes)))
    cs_accuracy=np.array((cs,len(cs_indexes)))
    cssb_accuracy=np.array((cssb,len(cssb_indexes)))
    nsb_accuracy=np.array((nsb,len(nsb_indexes)))
    csnsb_accuracy=np.array((csnsb,len(csnsb_indexes)))
    mb_accuracy=np.array((mb,len(mb_indexes)))

    if true_positives!=0:
        if(verbose=="True"): 
            print("Precision =",true_positives/(true_positives+false_positives))
            print("Recall =",true_positives/(true_positives+false_negatives))
            print(overall_accuracy,cs_accuracy,cssb_accuracy,nsb_accuracy,csnsb_accuracy,mb_accuracy,np.array((true_positives,false_positives,false_negatives)))

    return overall_accuracy,cs_accuracy,cssb_accuracy,nsb_accuracy,csnsb_accuracy,mb_accuracy,np.array((true_positives,false_positives,false_negatives))




   # np.array(true_positives,len(voiced_indexes)),np.array((cs,len(cs_indexes))),np.array((cssb,len(cssb_indexes))),np.array((nsb,len(nsb_indexes))),np.array((csnsb,len(csnsb_indexes))),np.array((mb,len(mb_indexes))),true_positives,false_positives,false_negatives
    
#ax2.set_ylabel('Y2-axis', color = 'black')

def get_arrays(label_list_1,len_frame_level,frame_len,frame_level):
    positives=np.zeros(len_frame_level)
    cleans=np.zeros(len_frame_level)
    cleans_backs=np.zeros(len_frame_level)
    nsb=np.zeros(len_frame_level)
    csnsb=np.zeros(len(frame_level))
    mb=np.zeros(len(frame_level))

   # print(len_frame_level)
    times=np.arange((len_frame_level))*(frame_len/1000)
    #print(times)
    for i in range(len(label_list_1)):
        current_label=label_list_1[i][2]
        previous_label=label_list_1[i-1][2]
        label_start=label_list_1[i][0]
        label_end=label_list_1[i][1]
        #print("CURRENT LABEL",current_label)
        
        if(current_label=="NSB"):
            if(previous_label=="CSNSB" or previous_label=="CS" or previous_label=="CSSSB"):
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)+1
                if(verbose=="True"): 
                    print(times[frame_start])

                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)

                if frame_end>=len_frame_level:
                    if(verbose=="True"): 
                        print(times[-1]+0.032)
                    nsb[frame_start:]=1
                else:
                    if(verbose=="True"): 
                        print(times[frame_end+1])
                nsb[frame_start:frame_end]=1
            else:   
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)
                if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
                if(verbose=="True"): 
                    print(times[frame_start])
                    print(times[frame_end])
                nsb[frame_start:frame_end]=1
        if(current_label=="MB"):
            if(previous_label=="CSNSB" or previous_label=="CS" or previous_label=="CSSSB"):
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)+1
                if(verbose=="True"): 
                    print(times[frame_start])

                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)
                if frame_end>=len_frame_level:
                    if(verbose=="True"): 
                        print(times[-1]+0.032)
                    mb[frame_start:]=1
                else:
                    if(verbose=="True"): 
                        print(times[frame_end+1])
                mb[frame_start:frame_end]=1
            else:
                if(verbose=="True"):    
                    print("Label END",label_end)
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)
                if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
                if(verbose=="True"): 
                    print("MB Start",times[frame_start])
                    print("MB End",times[frame_end])
                mb[frame_start:frame_end]=1

        if(current_label=="CSNSB"):
            if(previous_label=="CSNSB" or previous_label=="CS" or previous_label=="CSSSB"):
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)+1
                if(verbose=="True"): 
                    print(times[frame_start])

                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)
                if frame_end==len_frame_level:
                    if(verbose=="True"): 
                        print(times[-1]+0.032)
                    csnsb[frame_start:]=1
                else:
                    if(verbose=="True"): 
                        print(times[frame_end+1])
                csnsb[frame_start:frame_end]=1
            else:   
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)
                if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
                if(verbose=="True"): 
                    print(times[frame_start])
                    print(times[frame_end])
                csnsb[frame_start:frame_end+1]=1

        if(current_label=="CS" or current_label=="CSSB" or current_label=="CSNSB"):
            frame_start=float(label_start)/(frame_len/1000)
            frame_start=int(frame_start)
            frame_end=float(label_end)/(frame_len/1000)
            frame_end=int(frame_end)+1
            if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1
            positives[frame_start:frame_end+1]=1
            #print("Positives_array",positives)
            #print("FRAME START END",frame_start*(frame_len/1000),frame_end*(frame_len/1000))
        if(current_label=="CS"):
            frame_start=float(label_start)/(frame_len/1000)
            frame_start=int(frame_start)
            frame_end=float(label_end)/(frame_len/1000)
            frame_end=int(frame_end)+1
            if frame_end>=len_frame_level:
                    frame_end=len_frame_level-1

            cleans[frame_start:frame_end+1]=1


            if(verbose=="True"): 
                print("CS Start",times[frame_start])
                print("CS End",times[frame_end])

                print("CS start",cleans[frame_start-10:frame_start+1])
                print("CS end",cleans[frame_end:frame_end+10])
            #print("Positives_array",positives)
            #print("FRAME START END",frame_start*(frame_len/1000),frame_end*(frame_len/1000))
        if(current_label=="CSSB"):
            frame_start=float(label_start)/(frame_len/1000)
            frame_start=int(frame_start)
            frame_end=float(label_end)/(frame_len/1000)
            frame_end=int(frame_end)
            cleans_backs[frame_start:frame_end+1]=1
            if(verbose=="True"): 
                print("CSSB Start",times[frame_start])
                print("CSSB End",times[frame_end])
                #print("Positives_array",positives)
                #print("FRAME START END",frame_start*(frame_len/1000),frame_end*(frame_len/1000))
                print("Clean Backs start",cleans_backs[frame_start-10:frame_start+1])
                print("Clean Backs end",cleans_backs[frame_end:frame_end+10])




    if(verbose=="True"): 
        print("pos, cssb,csnsb,cs",get_num_ones(positives),get_num_ones(cleans_backs),get_num_ones(csnsb),get_num_ones(cleans))

    #print("From labels",positives)
    #print("Frame_level",frame_level)
    return positives,frame_level;

# def get_label_list(annotated_label=sys.argv[2],audio=sys.argv[1]):
def get_label_list(annotated_label,audio):

    frame_level=(probs(audio))
    #print(cont_frames(frame_level))
    final_result=cont_frames(frame_level)

    sr,data=scipy.io.wavfile.read(audio, mmap=False)
    #length=data.shape[0]/sr
    #time1=np.linspace(0.0,length,data.shape[0])
    #time2=np.linspace(0,(int(sys.argv[3])/1000)*len(frame_level),len(frame_level))

    #fig, ax1 = plt.subplots()

    list1=[]
    with open(annotated_label, mode ='r')as file:
       
      # reading the CSV file
      csvFile = csv.reader(file)
     
      # displaying the contents of the CSV file
      #print(type(csvFile))
      for lines in csvFile:
      		# k=(lines[0].split('\t'))
      		# print(k)
      		list1.append((lines[0].split('\t'))); 
      		#print("reached")
      #for lines in list1:
      		#print("reached here")
      		#print(lines)


    samplerate, data = wavfile.read(audio)
    #print("Checkpoint1",len(data))
    time=np.arange(len(data))/samplerate
    plt.plot(time,data)
    #plotter_func(list1,data)
    label_list=list1

    #return label_list,frame_level,final_result


    voiced_from_label,cleans,cleans_backs,nsb,csnsb,mb=frames_from_labels(label_list,len(final_result),int(sys.argv[1]),frame_level,audio)
    voiced_indexes=np.where(voiced_from_label==1)[0]
    not_speech_indexes=np.where(voiced_from_label==0)[0]
    cs_indexes=np.where(cleans==1)[0]
    cssb_indexes=np.where(cleans_backs==1)[0]
    nsb_indexes=np.where(nsb==1)[0]
    csnsb_indexes=np.where(csnsb==1)[0]
    mb_indexes=np.where(mb==1)[0]
    #print("HERE", cs_indexes)
    #Accuracy=get_accuracy(final_result,voiced_from_label,cs_indexes,cssb_indexes)
    #print(get_accuracy(final_result,voiced_indexes,cs_indexes,cssb_indexes,nsb_indexes))
    #print("Frame_level",frame_level)
    #print(get_accuracy(frame_level,voiced_indexes,not_speech_indexes,cs_indexes,cssb_indexes,nsb_indexes,csnsb_indexes,mb_indexes))
    return get_accuracy(frame_level,voiced_indexes,not_speech_indexes,cs_indexes,cssb_indexes,nsb_indexes,csnsb_indexes,mb_indexes)

def get_label_list_from_arrays(annotated_label,audio):

    frame_level=(probs(audio))
    #print(cont_frames(frame_level))
    final_result=cont_frames(frame_level)

    sr,data=scipy.io.wavfile.read(audio, mmap=False)
    #length=data.shape[0]/sr
    #time1=np.linspace(0.0,length,data.shape[0])
    #time2=np.linspace(0,(int(sys.argv[3])/1000)*len(frame_level),len(frame_level))

    #fig, ax1 = plt.subplots()

    list1=[]
    with open(annotated_label, mode ='r')as file:
       
      # reading the CSV file
      csvFile = csv.reader(file)
     
      # displaying the contents of the CSV file
      #print(type(csvFile))
      for lines in csvFile:
            # k=(lines[0].split('\t'))
            list1.append((lines[0].split('\t'))); 
      for lines in list1:
            print(lines)


    samplerate, data = wavfile.read(audio)
    #print("Checkpoint1",len(data))
    time=np.arange(len(data))/samplerate
    plt.plot(time,data)
    #plotter_func(list1,data)
    label_list=list1

    #return label_list,frame_level,final_result


    from_labels,vad_out=get_arrays(label_list,len(final_result),int(sys.argv[1]),frame_level)
    return from_labels,vad_out

from glob import glob

wavlist=glob("*.wav")
#overall_accuracy_1,cs_accuracy_1,cssb_accuracy_1,nsb_accuracy_1,csnsb_accuracy_1,mb_accuracy_1,results_1
Output=None
overall_accuracy_1=np.array([0,0])
cs_accuracy_1=np.array([0,0])
cssb_accuracy_1=np.array([0,0])
nsb_accuracy_1=np.array([0,0])
csnsb_accuracy_1=np.array([0,0])
mb_accuracy_1=np.array([0,0])
results_1=np.array([0,0,0])


for i in wavlist:
     print("\n \n",i)
     overall_accuracy,cs_accuracy,cssb_accuracy,nsb_accuracy,csnsb_accuracy,mb_accuracy,results=get_label_list(i[:-4]+".txt",i)
     overall_accuracy_1+=overall_accuracy
     cs_accuracy_1+=cs_accuracy
     cssb_accuracy_1+=cssb_accuracy
     nsb_accuracy_1+=nsb_accuracy
     csnsb_accuracy_1+=csnsb_accuracy
     mb_accuracy_1+=mb_accuracy
     results_1+=results



# for j in wavlist:
#      print("\n \n",j[:-4]+".csv",j)
#      Label_out,vad_out=get_label_list_from_arrays(j[:-4]+".csv",j)
#      plotter(j,Label_out,vad_out)
#      print("Label OUT",len(Label_out),Label_out)
#      print("VAD OUT",len(vad_out),vad_out)



print("Overall Accuracy",overall_accuracy_1,cs_accuracy_1,cssb_accuracy_1,nsb_accuracy_1,csnsb_accuracy_1,mb_accuracy_1,results_1)
print("CS Accuracy",cs_accuracy_1[0]/cs_accuracy_1[1])
print("CSSB Accuracy",cssb_accuracy_1[0]/cssb_accuracy_1[1])
print("CSNSB Accuracy",csnsb_accuracy_1[0]/csnsb_accuracy_1[1])
print("MB Accuracy",mb_accuracy_1[0]/mb_accuracy_1[1])
print("NSB Accuracy",nsb_accuracy_1[0]/nsb_accuracy_1[1])
print("Precision =",results_1[0]/(results_1[1]+results_1[0]))
print("Recall =",results_1[0]/(results_1[0]+results_1[2]))
precision=results_1[0]/(results_1[1]+results_1[0])
recall=results_1[0]/(results_1[0]+results_1[2])
print("F1",2*((precision*recall)/(precision+recall)))



# #frame_level=(probs(sys.argv[1]))
# print("\n")
# new1=(frame_level>0.75)
# A=np.zeros(len(new1))
# new1=list(new1)
# A[new1]=1
# print("A \n")
# print(A)
# print(frame_level)
# print(type(new1))
# print(cont_frames(A))
# final_result=cont_frames(frame_level)

# #sr,data=scipy.io.wavfile.read(sys.argv[1], mmap=False)
# length=data.shape[0]/sr
# time=np.linspace(0.0,length,data.shape[0])
# time2=np.arange(0,(10/1000)*len(frame_level),10/1000)

# fig, ax1 = plt.subplots()
# color = 'tab:blue'
# ax1.plot(time, data, color = color)

# ax2 = ax1.twinx()
 
# color = 'tab:red'
# #ax2.set_ylabel('Y2-axis', color = 'black')
# ax2.plot(time2,frame_level, color = color)
# ax2.plot(time2,final_result, color = "purple")
# ax2.tick_params(axis ='y', labelcolor = "black")

# plt.show()








        
