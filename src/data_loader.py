import os
import librosa
import numpy as np
from src.feature_extraction import extract_features

def load_data(data_dir):
    features = []
    labels = []

    data_dir = 'data/'

    
    # Iterate over each genre in the dataset folder
    for genre in os.listdir(data_dir):
        genre_dir = os.path.join(data_dir, genre)
        if os.path.isdir(genre_dir):
            # Iterate over each audio file in the genre folder
            for file_name in os.listdir(genre_dir):
                file_path = os.path.join(genre_dir, file_name)
                y, sr = librosa.load(file_path, duration=30)  # Load 30 seconds of audio
                feature = extract_features(y, sr)             # Extract features from the audio
                features.append(feature)                      # Append extracted features
                labels.append(genre)                          # Append the genre label
                
    return np.array(features), np.array(labels)
