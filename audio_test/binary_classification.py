import librosa
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)
  
def load_data(data_dir):
    features = []
    labels = []
    
    # Process each singer's folder
    for i, singer_folder in enumerate(sorted(os.listdir(data_dir))):
        folder_path = os.path.join(data_dir, singer_folder)
        
        if os.path.isdir(folder_path):
            print(f"Loading {singer_folder} samples...")
            
            for audio_file in os.listdir(folder_path):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(folder_path, audio_file)
                    voice_features = extract_features(file_path)
                    features.append(voice_features)
                    labels.append(i)  # 0 for first singer, 1 for second
    
    return np.array(features), np.array(labels)
  
data_dir = '/home/nirjhar/CODE/audio_test/dataset_5s'

# Load the voice data
print("Loading voice samples...")
X, y = load_data(data_dir)

print(f"Total samples: {len(X)}")
print(f"Features per sample: {X.shape[1]}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create and train SVM classifier
model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f} ({accuracy*100:.1f}%)")

# Show some predictions
print("\nSample predictions:")
for i in range(min(5, len(X_test))):
    actual = "Lata" if y_test[i] == 0 else "Arijit"
    predicted = "Lata" if y_pred[i] == 0 else "Arijit"
    print(f"Actual: {actual}, Predicted: {predicted}")