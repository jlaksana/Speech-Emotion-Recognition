import sounddevice as sd
from scipy.io.wavfile import write

def record_snippet(recording_number = 0):
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording
    print("Recording")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)

    sd.wait()  # Wait until recording is finished
    print("Done")
    write("new_sound%s.wav"%recording_number, fs, myrecording)  # Save as WAV file 