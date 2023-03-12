#importing the required packages
import numpy as np
from scipy.io import wavfile
import sys
from glob import glob


# Function to add 10dB SNR noise to a speech utterance. 
# Description = Clips a segment of noise from a random position from the noise file. 
# The noise segment is normalized and added to the speech. The noisy speech is saved to a separate file.
# Inputs:
# speechfile = Path to the speech utterance .wav file
# noisefile = Path to the noise .wav file
# outputfile = Path (along with a .wav file name) where the noisy speech file must be saved

def add_noise(speechfile, noisefile,SNR):
    
    #reading the .wav files
    sampFreq, noise = wavfile.read(noisefile)
    sampFreq, speech = wavfile.read(speechfile)
    numSamples = len(speech)

    #clipping a segment of noise from a random position, with segment length equal to the length of speech 
    i = np.random.choice(np.arange(len(noise) - numSamples))
    noise = noise[i:i+numSamples]

    #converting the PCM values to floats with range from -1.0 to 1.0
    speech = speech/32768
    noise = noise/32768

    #normalizing the noise and adding it to the speech
    rawEnergy = np.sum(speech**2)
    m_factor=10**(SNR/10)
    print("m_factor",(np.sqrt(rawEnergy/(m_factor*np.sum(noise**2)))))
    noise = noise*(np.sqrt(rawEnergy/(m_factor*np.sum(noise**2))))
    speech = speech + noise

    #normalizing the noisy speech so that its energy equals the energy of raw speech
    speech = speech*(np.sqrt(rawEnergy/np.sum(speech**2)))

    #converting the floats back to PCM values
    speech = speech*32767
    speech = speech.astype(np.int16)

    #saving the noisy speech to the output file
    #wavfile.write(outputfile, sampFreq, speech)
    
    return speech


def SNR_add(audio,SNR_val,k):
	
    print("Reached",audio)
    noisy_signal = add_noise(audio, "Noises/mixkit.wav", SNR_val)
    print("Reached",audio)
    wavfile.write(audio[:-4]+"_SB"+"_"+str(SNR_val)+"_"+str(k)+".wav", 16000, noisy_signal)

    noisy_signal = add_noise(audio, "Noises/wbrelaxing-rain-8228.wav", SNR_val)
    wavfile.write(audio[:-4]+"_EN"+"_"+str(SNR_val)+"_"+str(k)+".wav", 16000, noisy_signal)



    noisy_signal = add_noise(audio, "Noises/white_noise.wav", SNR_val)
    wavfile.write(audio[:-4]+"_WN"+"_"+str(SNR_val)+"_"+str(k)+".wav", 16000, noisy_signal)


if __name__ == "__main__":
    wavlist=glob("*.wav")
    
    #testing the function

    # speechfile = "./1b4c9b89_nohash_3.wav"
    # noisefile = "./running_tap.wav"
    # outputfile = "./noisy.wav"
    # add_noise(speechfile, noisefile, outputfile)

    for i in wavlist:
        for j in range(1,21):
        	for k in range(1,11):
             		SNR_add(i,j,k)
    # print("Reached",sys.argv[1])
    # noisy_signal = add_noise(sys.argv[1], "mixkit.wav", int(sys.argv[2]))
    # print("Reached",sys.argv[1])
    # wavfile.write(sys.argv[1][:-4]+"_SB.wav", 16000, noisy_signal)

    # noisy_signal = add_noise(sys.argv[1], "wbrelaxing-rain-8228.wav", int(sys.argv[2]))
    # wavfile.write(sys.argv[1][:-4]+"_EN.wav", 16000, noisy_signal)



    # noisy_signal = add_noise(sys.argv[1], "white_noise.wav", int(sys.argv[2]))
    # wavfile.write(sys.argv[1][:-4]+"_WN.wav", 16000, noisy_signal)
