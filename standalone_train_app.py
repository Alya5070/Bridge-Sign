from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import threading
import tensorflow as tf
from sklearn.model_selection import train_test_split

app = Flask(__name__)

base_dir = os.path.abspath(os.path.dirname(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
os.makedirs(dataset_dir, exist_ok=True)
csv_filepath = os.path.join(dataset_dir, 'asl_mediapipe_keypoints_dataset.csv')

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# State variables
is_recording = False
is_testing = False
current_label = ""
frames_recorded = 0
target_frames = 100

custom_model = None
custom_labels = []

def load_custom_model():
    global custom_model, custom_labels
    model_path = os.path.join(base_dir, 'Model', 'new_custom_model.h5')
    labels_path = os.path.join(base_dir, 'Model', 'new_custom_labels.txt')
    
    if os.path.exists(model_path) and os.path.exists(labels_path):
        custom_model = tf.keras.models.load_model(model_path)
        with open(labels_path, 'r') as f:
            custom_labels = [line.strip() for line in f.readlines()]
        return True
    return False

def get_normalized_landmarks(hand_landmarks):
    # Extract 21 points (x, y, z) relative to wrist (index 0)
    landmarks = []
    base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return landmarks

def generate_frames():
    global is_recording, is_testing, current_label, frames_recorded, target_frames, custom_model, custom_labels
    
    # Use CAP_DSHOW for faster startup on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while True:
        success, img = cap.read()
        if not success:
            continue
            
        # Flip image array for selfie view
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_img)
        
        pred_label = None
        pred_confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Physically draw the skeleton on the image 
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = get_normalized_landmarks(hand_landmarks)

                if is_recording and frames_recorded < target_frames:
                    # write to csv dataset
                    with open(csv_filepath, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        row = [current_label] + landmarks
                        writer.writerow(row)
                    
                    frames_recorded += 1
                    if frames_recorded >= target_frames:
                        is_recording = False

                elif is_testing and custom_model is not None and custom_labels:
                    # Predict live frame
                    try:
                        prediction = custom_model.predict(np.array([landmarks]), verbose=0)
                        class_id = np.argmax(prediction)
                        pred_confidence = prediction[0][class_id]
                        if pred_confidence > 0.5:
                            pred_label = custom_labels[class_id]
                    except Exception:
                        pass
        
        # Overlay UI instructions
        if is_recording:
            cv2.putText(img, f"Recording: {frames_recorded}/{target_frames}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_testing:
            if pred_label:
                cv2.putText(img, f"Sign: {pred_label} ({pred_confidence*100:.1f}%)", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(img, "Testing mode: Waiting for sign...", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(img, "Ready. Show sign to begin.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('standalone_trainer.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, is_testing, current_label, frames_recorded, target_frames
    data = request.json
    current_label = data.get('label', '').upper().strip()
    target_frames = int(data.get('frames', 100))
    
    if not current_label:
        return jsonify({"success": False, "error": "Label is required"})
        
    frames_recorded = 0
    is_testing = False # Stop testing if recording
    is_recording = True
    return jsonify({"success": True})

@app.route('/toggle_testing', methods=['POST'])
def toggle_testing():
    global is_testing, is_recording
    data = request.json
    action = data.get('action')
    
    if action == 'start':
        if load_custom_model():
            is_recording = False
            is_testing = True
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "No trained model found. Train the model first."})
    else:
        is_testing = False
        return jsonify({"success": True})

@app.route('/get_status')
def get_status():
    global is_recording, frames_recorded, target_frames
    return jsonify({
        "is_recording": is_recording,
        "frames_recorded": frames_recorded,
        "target_frames": target_frames
    })

# Training mechanism
is_training = False
training_log = []

def train_model_thread():
    global is_training, training_log
    
    try:
        training_log.append("Reading accumulated CSV dataset...")
        
        if not os.path.exists(csv_filepath):
            training_log.append("Error: Dataset CSV not found.")
            is_training = False
            return
            
        labels_list = []
        features_list = []
        with open(csv_filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 1:
                    # Skip header row if present
                    if row[0].lower() == 'label' or not row[1].replace('.', '', 1).replace('-', '', 1).isdigit():
                        continue
                    # Automatically detect if label is at the start (my format) or at the end (external project format)
                    try:
                        # Try to parse the last element as a float. If it fails, that means the last element is the label (e.g. 'A', 'B', 'HELLO')
                        float(row[-1])
                        
                        # If that succeeded, the label is at row[0]
                        label = row[0]
                        features = [float(x) for x in row[1:64]]
                    except ValueError:
                        # If it failed, the label is at row[-1]
                        label = row[-1]
                        features = [float(x) for x in row[0:63]]
                    
                    labels_list.append(label)
                    features_list.append(features)
                    
        if len(labels_list) == 0:
            training_log.append("Dataset is empty. Record data first.")
            return

        unique_labels = sorted(list(set(labels_list)))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        y = np.array([label_to_id[label] for label in labels_list])
        X = np.array(features_list)
        
        training_log.append(f"Loaded {len(X)} samples across {len(unique_labels)} signs: {', '.join(unique_labels)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        training_log.append("Building fast MLP Landmark Model...")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(unique_labels), activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        training_log.append("Training Model...")
        model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32, verbose=0)
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        training_log.append(f"Training Custom Model complete! Validation Accuracy: {acc*100:.2f}%")
        
        output_model_path = os.path.join(base_dir, 'Model', 'new_custom_model.h5')
        output_labels_path = os.path.join(base_dir, 'Model', 'new_custom_labels.txt')
        
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        model.save(output_model_path)
        
        with open(output_labels_path, 'w') as f:
            for lbl in unique_labels:
                f.write(f"{lbl}\n")
                
        training_log.append(f"Model successfully saved to {output_model_path}")
        training_log.append("Done. Ready for use!")
        
    except Exception as e:
        training_log.append(f"Error during training: {str(e)}")
        print(e)
    finally:
        is_training = False

@app.route('/train_model', methods=['POST'])
def train_model():
    global is_training, training_log
    if is_training:
        return jsonify({"success": False, "error": "Already training!"})
        
    is_training = True
    training_log = ["Starting training background process..."]
    t = threading.Thread(target=train_model_thread)
    t.start()
    return jsonify({"success": True})

@app.route('/training_status')
def training_status():
    global is_training, training_log
    return jsonify({
        "is_training": is_training,
        "log": training_log
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
