import os
import pandas as pandas
import numpy as numpy
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

def printAllSoundFiles(path):
    for dirname, _, filenames in os.walk(folderPath):
        for filename in filenames:
            print(os.path.join(dirname, filename))

def loadDatasets(path):
    """ 
        Label the audio file to its associated emotion. 
        paths[] stores all the audio file's path, labels[] stores the label of the audio file with the same index.
    """
    paths = []
    labels = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())

    paths.pop(0)
    labels.pop(0)

    return paths, labels

def createDataframe(paths, labels):
    """ Create dataframe from the given dataset """
    dataframe = pandas.DataFrame()
    dataframe['speech'] = paths
    dataframe['label'] = labels
    
    # print(dataframe.head())
    # print(dataframe['label'].value_counts())

    return dataframe

def showDataCountGraph(data):
    """ Display the number of data for each emotion """
    sns.countplot(data)
    plt.xlabel('Emotions')
    plt.ylabel('Data counts')
    plt.show()

def showWaveplot(data, sampleRate, emotion):
    """ Show the wave plot of a sound file """
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sampleRate)
    plt.show()

def showSpectogram(data, sampleRate, emotion):
    """ Show the spectogram of a sound file """
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))

    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sampleRate, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

def showWaveplotAndSpectogramForEmotion(dataframe, emotion):
    path = numpy.array(dataframe['speech'][dataframe['label']==emotion])[0]
    data, sampleRate = librosa.load(path)
    showWaveplot(data, sampleRate, emotion)
    showSpectogram(data, sampleRate, emotion)

def extractMFCC(filename):
    data, sampleRate = librosa.load(filename, duration=3, offset=0.5)
    mfcc = numpy.mean(librosa.feature.mfcc(y=data, sr=sampleRate, n_mfcc=40).T, axis=0)
    return mfcc

def extractMFCCfromAllFiles(dataframe):
    x_mfcc = dataframe['speech'].apply(lambda x: extractMFCC(x))
    x = [x for x in x_mfcc]
    x = numpy.array(x)
    x = numpy.expand_dims(x, -1)
    print(x.shape)

    enc = OneHotEncoder()
    y = enc.fit_transform(dataframe[['label']])
    y = y.toarray()
    print(y.shape)

def main():
    #printAllSoundFiles('./datasets')
    paths, labels = loadDatasets('./datasets')
    dataframe = createDataframe(paths, labels)
    # showDataCountGraph(dataframe['label'])
    # showWaveplotAndSpectogramForEmotion(dataframe, 'angry')
    extractMFCCfromAllFiles(dataframe)

main()
