# src/utils.py
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def encode_labels(labels):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_one_hot = to_categorical(labels_encoded)
    return labels_one_hot, label_encoder

def split_data(features, labels, test_size=0.2,  random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)
