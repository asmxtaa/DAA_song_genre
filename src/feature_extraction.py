import librosa
import numpy as np

def extract_features(y, sr):
    """
    Extract features from an audio file: MFCC, Chroma, Mel-spectrogram, and Zero Crossing Rate.
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Extract Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)
    
    # Extract Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    
    # Combine all features into a single array
    return np.hstack([mfccs_mean, chroma_mean, mel_mean, zcr_mean])
