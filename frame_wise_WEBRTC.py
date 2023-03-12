import collections
import contextlib
import sys
import wave
import pydub
import webrtcvad
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.io
import csv
verbose=(sys.argv[4])
def plotter_func(label_list,data):
  start_point=0;
  end_point=int(float(label_list[0][1])*16000)
  #print(start_point,end_point,end_point*1/16000)
  for i in range(len(label_list)):
    if(label_list[i][2]=="SIL"):
      label_color="grey";
    elif(label_list[i][2]=="CS"):
      label_color="red"
    elif(label_list[i][2]=="NSB"):
      label_color="magenta";
    else:
      label_color="green"
    time_1=np.arange(start_point,end_point+1)/16000

    #print("time_1",time_1[0],time_1[-1])
    #if i==len(label_list):
      #plt.plot(time_1,data[start_point:],color="red")
    plt.plot(time_1,data[start_point:end_point+1],color=label_color)
    if(i>=len(label_list)-2):
      start_point=end_point+1;
      end_point=len(data)-1;
      #print("time_1_1",time_1[-1])
    else:
      start_point=end_point+1;
      end_point=int(float(label_list[i+1][1])*16000)
      
      
def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames,voicing=0):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False
    #print("Total Frames",len(frames))
    Frame_level_result=np.zeros(len(frames))
    voiced_frames = []
    for frame in enumerate(frames):
        is_speech = vad.is_speech(frame[1].bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        Frame_level_result[frame[0]]=int(is_speech)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                #sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                #yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voicing=voicing+len(voiced_frames)
                voiced_frames = []
    #if triggered:
        #null
        #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if triggered:
       # yield b''.join([f.bytes for f in voiced_frames])
    	voicing=voicing+1
    return (voicing/len(frames)),Frame_level_result       #voicing/len(frames)


def percentage_voicing(args):
    # if len(args) != 2:
    #     sys.stderr.write('Usage: example.py <aggressiveness> <path to wav file>\n')
    #     sys.exit(1)
    audio, sample_rate = read_wave(args)
    vad = webrtcvad.Vad(3)
    frames = frame_generator(int(sys.argv[1]), audio, sample_rate)
    frames = list(frames)
    pervoice,result_array = vad_collector(sample_rate, int(sys.argv[1]), 300, vad, frames,0)
    #print(pervoice)
    #print(result_array)
    return pervoice,result_array

def cont_frames(frame_arr,threshold=0.99):
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
        if (trigger==False and (per>=0.9)):
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

def frames_from_labels(label_list_1,len_frame_level,frame_len,frame_level,audio,result_labels=(sys.argv[3]),verbose=(sys.argv[4])):
    positives=np.zeros(len_frame_level)
    cleans=np.zeros(len_frame_level)
    cleans_backs=np.zeros(len_frame_level)
    nsb=np.zeros(len_frame_level)
    csnsb=np.zeros(len(frame_level))
    mb=np.zeros(len(frame_level))

    if (result_labels=="True"):
        file1= open(audio[:-4]+"_frame_level_WEBRTC.txt","w")
    
    
    times=np.arange((len_frame_level))*(frame_len/1000)
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
                if (verbose=="True"):
                    print("NSB Start",times[frame_start])
                nsb_start=str(times[frame_start])

                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)

                if frame_end>=len_frame_level:
                    if (verbose=="True"):
                        print("NSB end",times[-1]+0.032)
                    nsb[frame_start:]=1
                    nsb_end=str(times[-1]+0.032)
                else:
                    if (verbose=="True"):
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
                if (verbose=="True"):
                    print("NSB Start",times[frame_start])
                nsb_start=str(times[frame_start])
                if (verbose=="True"):
                    print("NSB End",times[frame_end])
                nsb_end=str(times[frame_end])
                nsb[frame_start:frame_end]=1
                if (result_labels=="True"):
                    file1.write((nsb_start)+"\t\t"+(nsb_end)+"\t\t"+current_label+"\n")
        if(current_label=="MB"):
            if(previous_label=="CSNSB" or previous_label=="CS" or previous_label=="CSSB"):
                frame_start=float(label_start)/(frame_len/1000)
                frame_start=int(frame_start)+1
                if (verbose=="True"):
                    print("MB start",times[frame_start])
                mb_start=str(times[frame_start])

                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)
                if frame_end>=len_frame_level:
                    if (verbose=="True"):
                        print("MB start",times[-1]+0.01)
                    mb[frame_start:]=1
                    mb_end=str(times[-1]+0.01)
                else:
                    if (verbose=="True"):
                        print("MB END",times[frame_end+1])
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
                if (verbose=="True"):

                    print("MB Start",times[frame_start])
                mb_start=str(times[frame_start])
                if (verbose=="True"):
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
                if (verbose=="True"):
                    print("CSNB Start",times[frame_start])
                csnsb_start=str(times[frame_start])
                if (verbose=="True"):
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
                if (verbose=="True"):
                    print("CS Start",times[frame_start])
                cs_start=str(times[frame_start])
                if (verbose=="True"):
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
                if (verbose=="True"): 
                    print("CSSB Start",times[frame_start])
                cssb_start=str(times[frame_start])
                if (verbose=="True"):
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
                    if (verbose=="True"):
                        print("SIL Start",times[frame_start])
                    sil_start=str(times[frame_start])
                frame_end=float(label_end)/(frame_len/1000)
                frame_end=int(frame_end)

                if frame_end>=len_frame_level:
                    if (verbose=="True"):
                        print("SIL end",times[-1]+0.01)
                    nsb[frame_start:]=1
                    sil_end=str(times[-1]+0.01)
                else:
                    if (verbose=="True"):
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
                if (verbose=="True"):
                    print("SIL Start",times[frame_start])
                sil_start=str(times[frame_start])
                if (verbose=="True"):
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
    if (verbose=="True"):
        print("pos, cssb,csnsb,cs",get_num_ones(positives),get_num_ones(cleans_backs),get_num_ones(csnsb),get_num_ones(cleans))

        print("Frame level labeling",positives)
        print("Frame level Vad Output",frame_level)
    return positives,cleans,cleans_backs,nsb,csnsb,mb;

# def frames_from_labels(label_list_1,len_frame_level,frame_len,frame_level):
#     positives=np.zeros(len_frame_level)
#     cleans=np.zeros(len_frame_level)
#     cleans_backs=np.zeros(len_frame_level)
#     nsb=np.zeros(len_frame_level)
#     csnsb=np.zeros(len(frame_level))
#     mb=np.zeros(len(frame_level))

#     print(len_frame_level)
#     times=np.arange((len_frame_level))*(frame_len/1000)
#     #print(times)
#     for i in range(len(label_list_1)):
#         current_label=label_list_1[i][2]
#         previous_label=label_list_1[i-1][2]
#         label_start=label_list_1[i][0]
#         label_end=label_list_1[i][1]
#         #print("CURRENT LABEL",current_label)
        
#         if(current_label=="NSB"):
#             if(previous_label=="CSNSB" or previous_label=="CS" or previous_label=="CSSSB"):
#                 frame_start=float(label_start)/(frame_len/1000)
#                 frame_start=int(frame_start)+1
#                 print(times[frame_start])

#                 frame_end=float(label_end)/(frame_len/1000)
#                 frame_end=int(frame_end)

#                 if frame_end>=len_frame_level:
#                     print(times[-1]+0.030)
#                     nsb[frame_start:]=1
#                 else:
#                     print(times[frame_end+1])
#                 nsb[frame_start:frame_end]=1
#             else:   
#                 frame_start=float(label_start)/(frame_len/1000)
#                 frame_start=int(frame_start)
#                 frame_end=float(label_end)/(frame_len/1000)
#                 frame_end=int(frame_end)
#                 if frame_end>=len_frame_level:
#                     frame_end=len_frame_level-1
#                 print(times[frame_start])
#                 print(times[frame_end])
#                 nsb[frame_start:frame_end]=1
#         if(current_label=="MB"):
#             if(previous_label=="CSNSB" or previous_label=="CS" or previous_label=="CSSSB"):
#                 frame_start=float(label_start)/(frame_len/1000)
#                 frame_start=int(frame_start)+1
#                 print(times[frame_start])

#                 frame_end=float(label_end)/(frame_len/1000)
#                 frame_end=int(frame_end)
#                 if frame_end>=len_frame_level:
#                     print(times[-1]+0.030)
#                     mb[frame_start:]=1
#                 else:
#                     print(times[frame_end+1])
#                 mb[frame_start:frame_end]=1
#             else:   
#                 print("Label END",label_end)
#                 frame_start=float(label_start)/(frame_len/1000)
#                 frame_start=int(frame_start)
#                 frame_end=float(label_end)/(frame_len/1000)
#                 frame_end=int(frame_end)
#                 if frame_end>=len_frame_level:
#                     frame_end=len_frame_level-1
#                 print("MB Start",times[frame_start])
#                 print("MB End",times[frame_end])
#                 mb[frame_start:frame_end]=1

#         if(current_label=="CSNSB"):
#             if(previous_label=="CSNSB" or previous_label=="CS" or previous_label=="CSSSB"):
#                 frame_start=float(label_start)/(frame_len/1000)
#                 frame_start=int(frame_start)+1
#                 print(times[frame_start])

#                 frame_end=float(label_end)/(frame_len/1000)
#                 frame_end=int(frame_end)
#                 if frame_end==len_frame_level:
#                     print(times[-1]+0.030)
#                     csnsb[frame_start:]=1
#                 else:
#                     print(times[frame_end+1])
#                 csnsb[frame_start:frame_end]=1
#             else:   
#                 frame_start=float(label_start)/(frame_len/1000)
#                 frame_start=int(frame_start)
#                 frame_end=float(label_end)/(frame_len/1000)
#                 frame_end=int(frame_end)
#                 if frame_end>=len_frame_level:
#                     frame_end=len_frame_level-1
#                 print(times[frame_start])
#                 print(times[frame_end])
#                 csnsb[frame_start:frame_end+1]=1

#         if(current_label=="CS" or current_label=="CSSB" or current_label=="CSNSB"):
#             frame_start=float(label_start)/(frame_len/1000)
#             frame_start=int(frame_start)
#             frame_end=float(label_end)/(frame_len/1000)
#             frame_end=int(frame_end)+1
#             if frame_end>=len_frame_level:
#                     frame_end=len_frame_level-1
#             positives[frame_start:frame_end+1]=1
#             #print("Positives_array",positives)
#             #print("FRAME START END",frame_start*(frame_len/1000),frame_end*(frame_len/1000))
#         if(current_label=="CS"):
#             frame_start=float(label_start)/(frame_len/1000)
#             frame_start=int(frame_start)
#             frame_end=float(label_end)/(frame_len/1000)
#             frame_end=int(frame_end)+1
#             if frame_end>=len_frame_level:
#                     frame_end=len_frame_level-1

#             cleans[frame_start:frame_end+1]=1



#             print("CS Start",times[frame_start])
#             print("CS End",times[frame_end])

#             print("CS start",cleans[frame_start-10:frame_start+1])
#             print("CS end",cleans[frame_end:frame_end+10])
#             #print("Positives_array",positives)
#             #print("FRAME START END",frame_start*(frame_len/1000),frame_end*(frame_len/1000))
#         if(current_label=="CSSB"):
#             frame_start=float(label_start)/(frame_len/1000)
#             frame_start=int(frame_start)
#             frame_end=float(label_end)/(frame_len/1000)
#             frame_end=int(frame_end)
#             cleans_backs[frame_start:frame_end+1]=1

#             print("CSSB Start",times[frame_start])
#             print("CSSB End",times[frame_end])
#             #print("Positives_array",positives)
#             #print("FRAME START END",frame_start*(frame_len/1000),frame_end*(frame_len/1000))
#             print("Clean Backs start",cleans_backs[frame_start-10:frame_start+1])
#             print("Clean Backs end",cleans_backs[frame_end:frame_end+10])





#     print("pos, cssb,csnsb,cs",get_num_ones(positives),get_num_ones(cleans_backs),get_num_ones(csnsb),get_num_ones(cleans))

#     print("Frame level labeling",positives)
#     print("Frame level Vad Output",frame_level)
#     return positives,cleans,cleans_backs,nsb,csnsb,mb;

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

    overall_accuracy=np.array((true_positives,len(voiced_indexes)))
    cs_accuracy=np.array((cs,len(cs_indexes)))
    cssb_accuracy=np.array((cssb,len(cssb_indexes)))
    nsb_accuracy=np.array((nsb,len(nsb_indexes)))
    csnsb_accuracy=np.array((csnsb,len(csnsb_indexes)))
    mb_accuracy=np.array((mb,len(mb_indexes)))

    return overall_accuracy,cs_accuracy,cssb_accuracy,nsb_accuracy,csnsb_accuracy,mb_accuracy,np.array((true_positives,false_positives,false_negatives))



def get_label_list(annotated_label,audio):

    pervoice,frame_level=(percentage_voicing(audio))
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
        for lines in list1:
        #print("reached here")
            if (verbose=="True"):
                print(lines)


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
    if (verbose=="True"):
        print(get_accuracy(frame_level,voiced_indexes,not_speech_indexes,cs_indexes,cssb_indexes,nsb_indexes,csnsb_indexes,mb_indexes))
    return get_accuracy(frame_level,voiced_indexes,not_speech_indexes,cs_indexes,cssb_indexes,nsb_indexes,csnsb_indexes,mb_indexes)


#get_label_list(sys.argv[2],sys.argv[1])
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
    #print(i)
     if (verbose=="True"):
        print("\n \n Current file",i)
     overall_accuracy,cs_accuracy,cssb_accuracy,nsb_accuracy,csnsb_accuracy,mb_accuracy,results=get_label_list(i[:-4]+".txt",i)

     overall_accuracy_1+=overall_accuracy
     cs_accuracy_1+=cs_accuracy
     cssb_accuracy_1+=cssb_accuracy
     nsb_accuracy_1+=nsb_accuracy
     csnsb_accuracy_1+=csnsb_accuracy
     mb_accuracy_1+=mb_accuracy
     results_1+=results

print("Overall Accuracy",overall_accuracy_1,cs_accuracy_1,cssb_accuracy_1,nsb_accuracy_1,csnsb_accuracy_1,mb_accuracy_1,results_1)
print("Precision =",results_1[0]/(results_1[1]+results_1[0]))
print("Recall =",results_1[0]/(results_1[0]+results_1[2]))
print("CS identified as voiced=",((cs_accuracy_1[0]/cs_accuracy_1[1])*100))
print("CSSB identified as voiced=",((cssb_accuracy_1[0]/cssb_accuracy_1[1])*100))
print("NSB identified as voiced=",((nsb_accuracy_1[0]/nsb_accuracy_1[1])*100))
print("CSNSB identified as voiced=",((csnsb_accuracy_1[0]/csnsb_accuracy_1[1])*100))
print("MB identified as voiced=",((mb_accuracy_1[0]/mb_accuracy_1[1])*100))
print("%Voiced correctly identified",(cs_accuracy_1[0]+cssb_accuracy_1[0]+csnsb_accuracy_1[0])/(cs_accuracy_1[1]+cssb_accuracy_1[1]+csnsb_accuracy_1[1]))

precision=results_1[0]/(results_1[1]+results_1[0])
recall=results_1[0]/(results_1[0]+results_1[2])

print("F1",2*(precision*recall)/(precision+recall))




