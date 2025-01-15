
import numpy as np
import librosa
from keras import models as models

# Load the trained model
model = models.load_model('mymodel2_250.h5')

# Define a function to extract features from an audio file
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = 862 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None
    return mfccs

# Define a function to make predictions on an audio file
def make_prediction(file_name):
    features = extract_features(file_name)
    if features is not None:
        features = np.reshape(features, (*features.shape, 1))
        prediction = model.predict(np.array([features]))
        predicted_class = np.argmax(prediction)
        return predicted_class
    else:
        return None

# Test the inference script on an example audio file
file_name = 'example_audio.wav'
predicted_class = make_prediction(file_name)

if predicted_class is not None:
    class_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']
    print("Predicted class: ", class_names[predicted_class])
else:
    print("Error making prediction.")
