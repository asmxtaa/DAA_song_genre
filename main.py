from src.data_loader import load_data
from src.utils import encode_labels, split_data

# Load data and extract features
data_dir = 'data/'  # Path to the dataset
features, labels = load_data(data_dir)

# Encode the labels
labels_encoded, label_encoder = encode_labels(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(features, labels_encoded)

# Display the shape of training and testing data
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
