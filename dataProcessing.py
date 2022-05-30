import os
import pandas as pandas
import numpy as numpy
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from IPython.display import Audio
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, SimpleRNN
import warnings
warnings.filterwarnings('ignore')
import sounddevice as sd
from scipy.io.wavfile import write

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Pleasantly surprised', 'Sad']

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
    """ Show the waveplot and spectogram of a given emotion """
    path = numpy.array(dataframe['speech'][dataframe['label']==emotion])[0]
    data, sampleRate = librosa.load(path)
    showWaveplot(data, sampleRate, emotion)
    showSpectogram(data, sampleRate, emotion)

def extractMFCC(filename):
    """ Calculates the Mel-frequency cepstral coefficients of a wav file """
    data, sampleRate = librosa.load(filename, duration=3, offset=0.5)
    mfcc = numpy.mean(librosa.feature.mfcc(y=data, sr=sampleRate, n_mfcc=40).T, axis=0)
    return mfcc

def extractMFCCfromAllFiles(dataframe, enc):
    """ Calculates all the Mel-frequency cepstral coefficients of a dataframe of wav files """
    x_mfcc = dataframe['speech'].apply(lambda x: extractMFCC(x))
    x = [x for x in x_mfcc]
    x = numpy.array(x)
    x = numpy.expand_dims(x, -1)

    y = enc.fit_transform(dataframe[['label']])
    y = y.toarray()
    return x, y

def createLSTMModel():
    """Create a Long Short-term Model"""
    model = Sequential([
        LSTM(256, return_sequences=False, input_shape=(40,1)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def createFFNNModel():
    """Create a basic FeedForward Neural Network"""
    model = Sequential([
        Input(shape=(40,1)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def createRNNModel():
    """Create a Recurrent Neural Network"""
    model = Sequential([
        Input(shape=(40,1)),
        SimpleRNN(256, return_sequences=True, activation='relu'),
        SimpleRNN(256, activation='relu'),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def trainModel(data, labels, model):
    """ Fits a given model and displays its history """
    history = model.fit(data, labels, validation_split=0.2, epochs=5, batch_size=64)
    print(history)

def getPredictedEmotion(prediction_result):
    """ Given an array of predictions, returns the predicted emotion and confidence """
    highest_prediction_index = numpy.argmax(prediction_result)
    return EMOTIONS[highest_prediction_index], prediction_result[highest_prediction_index]

def recordSound():
    """ When called, uses the user's microphone to record for three seconds """
    freq = 44100
    duration = 3
    print("Recording...")
    recording = sd.rec(int(duration*freq), samplerate=freq, channels=1)
    sd.wait()
    print("Finished recording...")
    write("recording0.wav", freq, recording)

def extractRecordedSound(filename):
    """ Given the recorded filename, returns a numpy array of the Mel-frequency cepstral coefficients """
    mfcc = extractMFCC(filename)
    x = [mfcc]
    x = numpy.array(x)
    x = numpy.expand_dims(x, -1)
    return x

def saveModel(model, model_name):
    """ Saves a given pre-trained model into the saved_models folder """
    print('saving model...')
    model.save('saved_models/' + model_name)
    print('finished saving model')

def loadModel(model_name):
    """ Loads a selected pre-trained model and returns the model """
    print('loading model...')
    model = tf.keras.models.load_model('saved_models/' + model_name)
    # model = tf.keras.models.load_model('saved_models/nn_model')
    #model = tf.keras.models.load_model('saved_models/rnn_model')
    model.summary()
    print('finished loading model')
    return model

def main():
    encoder = OneHotEncoder()

    # printAllSoundFiles('./datasets')
    paths, labels = loadDatasets('./datasets')
    dataframe = createDataframe(paths, labels)
    # showDataCountGraph(dataframe['label'])
    # showWaveplotAndSpectogramForEmotion(dataframe, 'angry')
    df_train, df_test = train_test_split(dataframe, test_size=0.2)
    # x_train, y_train = extractMFCCfromAllFiles(df_train, encoder)
    x_test, y_test = extractMFCCfromAllFiles(df_test, encoder)

    # Create and train model
    # model = createLSTMModel()
    # model = createFFNNModel()
    # model = createRNNModel();
    # trainModel(x_train, y_train, model)

    # model.save('saved_models/lstm_model')
    # model.save('saved_models/nn_model')
    # model.save('saved_models/rnn_model')
    model_name = "lstm_model"
    loadedModel = loadModel(model_name)

    test_loss, test_acc = loadedModel.evaluate(x_test, y_test, verbose=2)

    print('\nTest accuracy:', test_acc)

    idx = 0
    prediction = loadedModel.predict(x_test)
    prediction_result = prediction[idx]
    print("Predicted:", getPredictedEmotion(prediction_result))
    print("Expected:",getPredictedEmotion(y_test[idx]))

    # Recorded audio
    # recordSound()
    # filename = "recording0.wav"
    # print("Predicting recorded audio:")
    # result = extractRecordedSound(filename)
    # prediction = loadedModel.predict(result)
    # print("Prediction result:", getPredictedEmotion(prediction[0])[0])
    # print("Confidence result:", getPredictedEmotion(prediction[0])[1])

if __name__ == '__main__':
    main()