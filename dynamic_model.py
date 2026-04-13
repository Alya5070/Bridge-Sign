import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

base_dir = os.path.abspath(os.path.dirname(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
dynamic_csv_filepath = os.path.join(dataset_dir, 'dynamic_point_history.csv')
model_save_path = os.path.join(base_dir, 'dynamic_custom_model.h5')
# We need to save labels for dynamic model somewhere, let's use dynamic_labels.json
labels_save_path = os.path.join(base_dir, 'dynamic_labels.json')

def train_dynamic_model(training_log=None):
    def log(msg):
        print(msg)
        if training_log is not None:
            training_log.append(msg)

    if not os.path.exists(dynamic_csv_filepath):
        log(f"Dataset not found at {dynamic_csv_filepath}. Please record dynamic gestures first in the Trainer Module.")
        return False

    X = []
    Y_labels = []

    try:
        with open(dynamic_csv_filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                label = row[0]
                features = [float(val) for val in row[1:]]
                if len(features) == 64: # 32 for primary + 32 for secondary
                    X.append(features)
                    Y_labels.append(label)
    except Exception as e:
        log(f"Error parsing dataset: {e}")
        return False

    if len(X) < 10:
        log("Not enough data to train. Please collect more dynamic gestures.")
        return False

    unique_labels = sorted(list(set(Y_labels)))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    
    Y_indices = [label_to_index[label] for label in Y_labels]
    
    X = np.array(X)
    Y = to_categorical(Y_indices, num_classes=len(unique_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(64,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    log(f"Training dynamic model on {len(X_train)} samples across {len(unique_labels)} classes...")
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    log(f"Model Accuracy on Test Set: {accuracy*100:.2f}%")

    model.save(model_save_path)
    
    import json
    with open(labels_save_path, 'w') as f:
        json.dump(unique_labels, f)

    log(f"Dynamic Model saved to {model_save_path}")
    log(f"Labels saved to {labels_save_path}")
    return True

if __name__ == '__main__':
    train_dynamic_model()
