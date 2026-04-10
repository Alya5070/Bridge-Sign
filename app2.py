from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, flash
import cv2
import numpy as np
import math
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import shutil
import csv
import json
import threading
from datetime import datetime
import re

# --- MACHINE LEARNING & TRAINER IMPORTS ---
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def add_header(response):
    """
    Add headers to force browser to not cache templates
    to ensure the new UI glassmorphism design displays.
    """
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Configure SQLite Database (No external server required)
base_dir = os.path.abspath(os.path.dirname(__file__))
instance_dir = os.path.join(base_dir, 'instance')
os.makedirs(instance_dir, exist_ok=True)
db_path = os.path.join(instance_dir, 'msl_app.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Dataset path for Trainer Module
dataset_dir = os.path.join(base_dir, 'dataset')
os.makedirs(dataset_dir, exist_ok=True)
csv_filepath = os.path.join(dataset_dir, 'asl_mediapipe_keypoints_dataset.csv')

# MediaPipe hands setup for Trainer Module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def get_normalized_landmarks(hand_landmarks, flip_x=False):
    # Extract 21 points (x, y, z) relative to wrist (index 0)
    landmarks = []
    base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
    for lm in hand_landmarks.landmark:
        x_val = lm.x - base_x
        if flip_x:
            x_val = -x_val  # Mirror the X coordinate
        landmarks.extend([x_val, lm.y - base_y, lm.z - base_z])
    return landmarks


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    camera_enabled = db.Column(db.Boolean, default=True)
    camera_index = db.Column(db.Integer, default=0)
    is_admin = db.Column(db.Boolean, default=False)
    tutorial_completed = db.Column(db.Boolean, default=False)
    security_question = db.Column(db.String(200), nullable=True)
    security_answer_hash = db.Column(db.String(200), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def set_security_answer(self, answer):
        self.security_answer_hash = generate_password_hash(answer.strip().lower())

    def check_security_answer(self, answer):
        if not self.security_answer_hash:
            return False
        return check_password_hash(self.security_answer_hash, answer.strip().lower())

class WordRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    requested_word = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default="Pending") # Pending, Added, Rejected
    suggestion_type = db.Column(db.String(50), default="ai_recognition") # ai_recognition or text_to_sign
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('requests', lazy=True))

with app.app_context():
    db.create_all()
    # Schema Migration for newly added columns
    try:
        db.session.execute(db.text('ALTER TABLE user ADD COLUMN security_question VARCHAR(200)'))
        db.session.execute(db.text('ALTER TABLE user ADD COLUMN security_answer_hash VARCHAR(200)'))
        db.session.commit()
    except Exception:
        db.session.rollback()
        pass
    
    try:
        db.session.execute(db.text('ALTER TABLE word_request ADD COLUMN suggestion_type VARCHAR(50) DEFAULT "ai_recognition"'))
        db.session.commit()
    except Exception:
        db.session.rollback()
        pass
    # Ensure Admin user exists
    admin_user = User.query.filter_by(username='Admin').first()
    if not admin_user:
        admin_user = User(username='Admin', is_admin=True)
        admin_user.set_password('admin123')
        db.session.add(admin_user)
        db.session.commit()
    elif not admin_user.is_admin:
        admin_user.is_admin = True
        db.session.commit()


# Initialize variables and models
base_dir = os.path.abspath(os.path.dirname(__file__))
active_mode = "spelling"  # Default mode

# Initialize variables and models
base_dir = os.path.abspath(os.path.dirname(__file__))
active_mode = "spelling"  # Default mode

classifier = None
labels = []

def load_model(mode):
    global active_mode
    active_mode = mode
    
    # We natively bridge the legacy route to our active custom model now
    success = load_custom_model()
    if success:
        print(f"✅ Successfully piped {mode} routing to MediaPipe new_custom_model.h5")
        return True
    else:
        print(f"❌ Error loading {mode} model! Custom model not found.")
        return False



# Global variables
offset = 30
imgSize = 300
current_prediction = ""
confidence_scores = []
practice_word = ""
current_letter_index = 0
practice_active = False

# Global camera variables
camera_active = True
current_camera_index = 0

# Trainer Module Globals
camera_mode = 'training_idle'  # spelling, words, training_idle, training_record, testing
is_recording = False
is_testing = False
is_training = False
current_label = ""
frames_recorded = 0
target_frames = 100
custom_model = None
custom_labels = []
training_log = []

# Camera caching
cached_cameras = []

def get_camera_list():
    global cached_cameras
    
    # If we already have a cached list and it's not empty, return it
    if cached_cameras:
        return cached_cameras
        
    print("Discovering available cameras via DirectShow...")
    available_cameras = []
    
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        
        # Filter out obvious non-camera or duplicate virtual interfaces
        invalid_keywords = ['DFU', 'Virtual', 'Control', 'Render']
        
        for idx, name in enumerate(devices):
            is_valid = True
            for kw in invalid_keywords:
                if kw.lower() in name.lower():
                    is_valid = False
                    break
                    
            if is_valid:
                available_cameras.append({'index': idx, 'name': name})
                    
    except ImportError:
        print("Warning: pygrabber not installed. Falling back to basic check.")
        # Fallback to basic indexing if pygrabber isn't there
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    available_cameras.append({'index': i, 'name': f"Camera {i}"})
                    cap.release()
            except Exception:
                pass
    except Exception as e:
        print(f"Error enumerating cameras: {e}")
        
    # Cache and return
    cached_cameras = available_cameras
    print(f"Found {len(cached_cameras)} cameras: {cached_cameras}")
    return cached_cameras


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


# Global camera object and lock for thread safety
cap = None
camera_lock = threading.Lock()
hands_lock = threading.Lock()
active_index = -1
def generate_frames():
    global current_prediction, confidence_scores, camera_active, current_camera_index
    global camera_mode, is_recording, is_testing, current_label, frames_recorded, target_frames, custom_model, custom_labels
    global cap, active_index
    
    dark_frame_count = 0
    
    while True:
        try:
            # Check if camera should be active and reinitialize if needed or if camera index changed
            if camera_active and (cap is None or not cap.isOpened() or active_index != current_camera_index):
                # Immediate yield to tell browser we are starting if not already
                if cap is None:
                    status_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(status_frame, "Initializing Camera Stream...", (120, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', status_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                with camera_lock:
                    # Re-check inside lock to avoid double-init
                    if cap is None or not cap.isOpened() or active_index != current_camera_index:
                        if cap is not None:
                            cap.release()
                            cap = None
                            
                        print(f"Switching/Initializing camera to index {current_camera_index}...")
                        # 1. Try default backend first
                        cap = cv2.VideoCapture(current_camera_index)
                        
                        # 2. Try with CAP_DSHOW as fallback
                        if cap is None or not cap.isOpened():
                            cap = cv2.VideoCapture(current_camera_index, cv2.CAP_DSHOW)
                        
                        if cap.isOpened():
                            active_index = current_camera_index
                            # Small delay to let the driver stabilize
                            threading.Event().wait(0.5)
                            
                        # 3. Try Auto-Discovery on other indices if current fails
                        if cap is None or not cap.isOpened():
                            if cached_cameras:
                                for cam in cached_cameras:
                                    if cam['index'] != current_camera_index:
                                        cap = cv2.VideoCapture(cam['index'])
                                        if cap.isOpened():
                                            current_camera_index = cam['index']
                                            active_index = current_camera_index
                                            break
                
                if cap is None or not cap.isOpened():
                    # Return a black frame with error message
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_frame, "Camera not available", (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', black_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    threading.Event().wait(1.0) # wait before retrying
                    continue
                else:
                    # Warm up: skip first few frames which might be black/blurry on some cameras
                    with camera_lock:
                        if cap is not None and cap.isOpened():
                            for _ in range(10): # Increased warm up
                                cap.read()

            if not camera_active:
                with camera_lock:
                    if cap is not None:
                        cap.release()
                        cap = None

                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "Camera Disabled", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                threading.Event().wait(1.0)
                continue

            # Thread-safe read
            success = False
            img = None
            with camera_lock:
                if cap is not None and cap.isOpened():
                    success, img = cap.read()
            
            if not success or img is None:
                # If read fails, maybe try to re-init in next loop
                threading.Event().wait(0.1)
                continue

            # Ensure img is a valid numpy array before copy
            if not isinstance(img, np.ndarray):
                continue

            imgOutput = img.copy()

            # ---------------------------------------------
            # BRANCH 1: NORMAL TRANSLATION MODE (MediaPipe Custom)
            # ---------------------------------------------
            if camera_mode == 'translation':
                imgOutput = cv2.flip(imgOutput, 1) # Mirroring for user friendliness
                rgb_img = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
                
                with hands_lock:
                    results = hands.process(rgb_img)
                
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_type = handedness.classification[0].label
                        
                        # Mirror Right Hand coords to Left Hand geometry for standard detection
                        flip_x = (hand_type == 'Right')
                        
                        mp_drawing.draw_landmarks(imgOutput, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        landmarks = get_normalized_landmarks(hand_landmarks, flip_x=flip_x)
                        
                        if custom_model is not None and len(custom_labels) > 0:
                            try:
                                # FAST INFERENCE: Using __call__ directly on tensor avoids the .predict() callback overhead
                                input_tensor = tf.convert_to_tensor([landmarks])
                                prediction = custom_model(input_tensor, training=False).numpy()
                                class_id = np.argmax(prediction)
                                pred_confidence = prediction[0][class_id]
                                
                                if pred_confidence > 0.5:
                                    current_prediction = custom_labels[class_id]
                                    confidence_scores = prediction[0].tolist()
                                    cv2.putText(imgOutput, f"{current_prediction} ({pred_confidence*100:.1f}%)", (50, 50),
                                                cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
                                else:
                                    current_prediction = "Low confidence"
                                    confidence_scores = []
                            except Exception as e:
                                print(f"Error predicting: {e}")
                                current_prediction = "Prediction Error"
                                confidence_scores = []
                        else:
                            current_prediction = "Model not loaded"
                            confidence_scores = []
                            cv2.putText(imgOutput, "Model not loaded. Train in Trainer Module.", (20, 50),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    current_prediction = "No hand detected"
                    confidence_scores = []
                    
            # ---------------------------------------------
            # BRANCH 2: TRAINING / TESTING MODE (MediaPipe)
            # ---------------------------------------------
            elif camera_mode in ['training_idle', 'testing']:
                imgOutput = cv2.flip(imgOutput, 1) # Selfie view for training
                rgb_img = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
                
                with hands_lock:
                    results = hands.process(rgb_img)
                
                pred_label = None
                pred_confidence = 0.0

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_type = handedness.classification[0].label
                        
                        # Standardize all data to Left Hand geometry by mirroring Right hands
                        flip_x = (hand_type == 'Right')
                        
                        mp_drawing.draw_landmarks(imgOutput, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        landmarks = get_normalized_landmarks(hand_landmarks, flip_x=flip_x)

                        if is_recording and frames_recorded < target_frames:
                            with open(csv_filepath, mode='a', newline='') as f:
                                writer = csv.writer(f)
                                row = [current_label] + landmarks
                                writer.writerow(row)
                            
                            frames_recorded += 1
                            if frames_recorded >= target_frames:
                                is_recording = False

                        elif is_testing and custom_model is not None and custom_labels:
                            try:
                                # FAST INFERENCE
                                input_tensor = tf.convert_to_tensor([landmarks])
                                prediction = custom_model(input_tensor, training=False).numpy()
                                class_id = np.argmax(prediction)
                                pred_confidence = prediction[0][class_id]
                                if pred_confidence > 0.5:
                                    pred_label = custom_labels[class_id]
                            except Exception:
                                pass
                
                # Overlay UI instructions over the training feed
                if is_recording:
                    cv2.putText(imgOutput, f"Recording: {frames_recorded}/{target_frames}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif is_testing:
                    if pred_label:
                        cv2.putText(imgOutput, f"Sign: {pred_label} ({pred_confidence*100:.1f}%)", (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        cv2.putText(imgOutput, "Testing mode: Waiting for sign...", (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(imgOutput, "Ready. Show sign to begin.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # --- DEBUG OVERLAY ---
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(imgOutput, f"Stream: {timestamp} Mode: {camera_mode}", (10, imgOutput.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Check for black frame
            # Check for black frame
            if np.mean(imgOutput) < 5:
                dark_frame_count += 1
                if dark_frame_count > 30: # Only show warning after ~1 second of darkness
                    cv2.putText(imgOutput, "DARK FRAME DETECTED - CHECK CAMERA", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                dark_frame_count = 0

            # Encode unified frame
            ret, buffer = cv2.imencode('.jpg', imgOutput)
            frame = buffer.tobytes()

            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"ERROR IN GENERATE_FRAMES: {e}")
                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img, f"STREAM ERROR: {str(e)[:40]}", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
        except GeneratorExit:
            raise
        except Exception as global_e:
            print(f"CRITICAL GENERATOR ERROR: {global_e}")
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "CRITICAL ERROR", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.context_processor
def inject_user():
    context = {'username': '', 'is_admin': False}
    if 'user_id' in session:
        user = db.session.get(User, session['user_id'])
        if user:
            context['username'] = user.username
            context['is_admin'] = user.is_admin
    return context

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        security_question = request.form.get('security_question', '').strip()
        security_answer = request.form.get('security_answer', '').strip()

        # Validation
        if not username or not password or not security_question or not security_answer:
            flash('All fields are required', 'error')
            return render_template('register.html')

        if len(username) < 3:
            flash('Username must be at least 3 characters long', 'error')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')

        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'error')
            return render_template('register.html')

        # Create new user
        new_user = User(username=username, security_question=security_question)
        new_user.set_password(password)
        new_user.set_security_answer(security_answer)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'error')
            return render_template('register.html')

    return render_template('register.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if 'user_id' in session:
        return redirect(url_for('index'))

    step = 1
    username = ''
    question = ''

    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username', '').strip()

        if action == 'verify_user':
            user = User.query.filter_by(username=username).first()
            if not user:
                flash('User not found.', 'error')
            elif not user.security_question:
                flash('This account does not have a security question set up. Please contact an admin.', 'error')
            else:
                step = 2
                question = user.security_question

        elif action == 'reset_password':
            user = User.query.filter_by(username=username).first()
            if not user:
                flash('User not found.', 'error')
                return render_template('forgot_password.html', step=1)

            answer = request.form.get('security_answer', '')
            new_password = request.form.get('new_password', '')
            confirm_password = request.form.get('confirm_password', '')

            if not user.check_security_answer(answer):
                flash('Incorrect security answer.', 'error')
                step = 2
                question = user.security_question
            elif len(new_password) < 6:
                flash('New password must be at least 6 characters.', 'error')
                step = 2
                question = user.security_question
            elif new_password != confirm_password:
                flash('Passwords do not match.', 'error')
                step = 2
                question = user.security_question
            else:
                user.set_password(new_password)
                db.session.commit()
                flash('Password reset successfully. Please login with your new password.', 'success')
                return redirect(url_for('login'))

    return render_template('forgot_password.html', step=step, username=username, question=question)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and not user.tutorial_completed:
            return redirect(url_for('tutorial'))
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            if not user.tutorial_completed:
                return redirect(url_for('tutorial'))
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def index():
    global camera_active, current_camera_index, camera_mode
    user = User.query.get(session['user_id'])

    if user is None:
        session.clear()
        flash('Session expired. Please login again.', 'error')
        return redirect(url_for('login'))

    if not user.tutorial_completed:
        return redirect(url_for('tutorial'))

    camera_mode = 'translation'
    
    # Preload cameras on dashboard load to populate cache
    get_camera_list()

    camera_active = user.camera_enabled
    current_camera_index = user.camera_index
    return render_template('dashboard.html', username=session['username'], camera_active=camera_active, is_admin=user.is_admin)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.get(session['user_id'])

    if user is None:
        session.clear()
        flash('Session expired. Please login again.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_username = request.form.get('username', '').strip()
        security_question = request.form.get('security_question', '').strip()
        security_answer = request.form.get('security_answer', '').strip()
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not current_password:
            flash('Current password is required to make changes', 'error')
            return render_template('profile.html', user=user)
            
        if not user.check_password(current_password):
            flash('Incorrect current password', 'error')
            return render_template('profile.html', user=user)

        if not new_username:
            flash('Username cannot be empty', 'error')
            return render_template('profile.html', user=user)

        if len(new_username) < 3:
            flash('Username must be at least 3 characters long', 'error')
            return render_template('profile.html', user=user)

        # Check if username is taken by another user
        existing_user = User.query.filter_by(username=new_username).first()
        if existing_user and existing_user.id != user.id:
            flash('Username already taken', 'error')
            return render_template('profile.html', user=user)

        if new_password:
            if len(new_password) < 6:
                flash('New password must be at least 6 characters long', 'error')
                return render_template('profile.html', user=user)
            if new_password != confirm_password:
                flash('New passwords do not match', 'error')
                return render_template('profile.html', user=user)
            user.set_password(new_password)

        user.username = new_username
        session['username'] = new_username
        
        if security_question and security_answer:
            user.security_question = security_question
            user.set_security_answer(security_answer)

        try:
            db.session.commit()
            flash('Profile updated successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'error')

    return render_template('profile.html', user=user)


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    global camera_active, current_camera_index
    user = User.query.get(session['user_id'])

    if user is None:
        session.clear()
        flash('Session expired. Please login again.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'toggle_camera':
            user.camera_enabled = not user.camera_enabled
            camera_active = user.camera_enabled
            db.session.commit()
            status = "enabled" if camera_active else "disabled"
            flash(f'Camera {status} successfully!', 'success')

        elif action == 'change_camera':
            new_index = int(request.form.get('camera_index', 0))
            user.camera_index = new_index
            current_camera_index = new_index
            db.session.commit()
            flash(f'Camera switched to index {new_index}', 'success')

    # Get available cameras safely with names
    available_cameras = get_camera_list()

    return render_template('settings.html',
                           user=user,
                           available_cameras=available_cameras)


@app.route('/refresh_cameras', methods=['POST'])
@login_required
def refresh_cameras():
    global cached_cameras
    cached_cameras = [] # Clear cache
    get_camera_list() # Repopulate cache
    flash('Camera list refreshed.', 'success')
    return redirect(url_for('settings'))


@app.route('/tutorial')
@login_required
def tutorial():
    global camera_active, current_camera_index
    user = db.session.get(User, session['user_id'])
    
    if user is None:
        session.clear()
        flash('Session expired. Please login again.', 'error')
        return redirect(url_for('login'))

    camera_active = user.camera_enabled
    current_camera_index = user.camera_index
    return render_template('tutorial.html', username=session['username'], camera_active=camera_active, is_admin=user.is_admin)


@app.route('/quiz')
@login_required
def quiz():
    global camera_active, current_camera_index, camera_mode
    user = db.session.get(User, session['user_id'])
    if user is None:
        session.clear()
        return redirect(url_for('login'))
        
    camera_active = user.camera_enabled
    current_camera_index = user.camera_index
    camera_mode = 'translation'
    
    # Preload cameras on quiz load to populate cache if needed
    get_camera_list()
    
    return render_template('quiz.html', username=session.get('username', ''), camera_active=camera_active, is_admin=user.is_admin)


@app.route('/complete_tutorial', methods=['POST'])
@login_required
def complete_tutorial():
    user = db.session.get(User, session['user_id'])
    if user:
        user.tutorial_completed = True
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False}), 400



@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_prediction')
@login_required
def get_prediction():
    global current_letter_index, practice_active

    safe_confidence = [float(score) for score in confidence_scores] if confidence_scores else []

    should_advance = False
    if practice_active and practice_word and current_letter_index < len(practice_word):
        # Allow case-insensitive matching for both letters and words
        expected_letter = practice_word[current_letter_index].upper()
        if current_prediction and current_prediction.upper() == expected_letter:
            should_advance = True

    return jsonify({
        'prediction': current_prediction,
        'confidence': safe_confidence,
        'practice_active': practice_active,
        'practice_word': practice_word,
        'current_letter_index': current_letter_index,
        'should_advance': should_advance,
        'active_mode': active_mode
    })


@app.route('/switch_mode', methods=['POST'])
@login_required
def switch_mode():
    data = request.json
    new_mode = data.get('mode', 'spelling')
    
    if new_mode not in ['spelling', 'words']:
        return jsonify({'success': False, 'error': 'Invalid mode'}), 400
        
    success = load_model(new_mode)
    return jsonify({
        'success': success, 
        'mode': active_mode,
        'labels': labels
    })


@app.route('/get_labels')
@login_required
def get_labels():
    mode = request.args.get('mode', active_mode)
    curr_active = active_mode
    
    # Temporarily switch to read labels if different
    if mode != active_mode:
        load_model(mode)
        resp_labels = labels.copy()
        load_model(curr_active) # switch back
        return jsonify({'labels': resp_labels})
    
    return jsonify({'labels': labels})


@app.route('/set_practice_word', methods=['POST'])
@login_required
def set_practice_word():
    global practice_word, current_letter_index, practice_active
    data = request.get_json()
    practice_word = data.get('word', '').upper()
    current_letter_index = 0
    practice_active = False
    return jsonify({'success': True, 'word': practice_word})


@app.route('/start_practice', methods=['POST'])
@login_required
def start_practice():
    global practice_active, current_letter_index
    practice_active = True
    current_letter_index = 0
    return jsonify({'success': True})


@app.route('/reset_practice', methods=['POST'])
@login_required
def reset_practice():
    global practice_word, current_letter_index, practice_active
    practice_word = ""
    current_letter_index = 0
    practice_active = False
    return jsonify({'success': True})


@app.route('/advance_letter', methods=['POST'])
@login_required
def advance_letter():
    global current_letter_index
    if current_letter_index < len(practice_word) - 1:
        current_letter_index += 1
    return jsonify({'success': True, 'current_index': current_letter_index})


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login as admin to continue.', 'error')
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            flash('Admin access required.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


# ==========================================
# TRAINER MODULE API ENDPOINTS (ADMIN ONLY)
# ==========================================

def load_custom_model():
    global custom_model, custom_labels, labels
    model_path = os.path.join(base_dir, 'Model', 'new_custom_model.h5')
    labels_path = os.path.join(base_dir, 'Model', 'new_custom_labels.txt')
    
    if os.path.exists(model_path) and os.path.exists(labels_path):
        # Prevent memory leaks when reloading models repeatedly
        tf.keras.backend.clear_session()
        custom_model = tf.keras.models.load_model(model_path)
        with open(labels_path, 'r') as f:
            custom_labels = [line.strip() for line in f.readlines()]
            labels = custom_labels
        return True
    return False

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
                # Must have at least a label and one set of coordinates (e.g. 63 values)
                if len(row) < 5:
                    continue
                    
                # Skip header row if present
                is_header = False
                if row[0].lower() == 'label':
                    is_header = True
                else:
                    try:
                        float(row[1])
                    except ValueError:
                        is_header = True
                
                if is_header:
                    continue
                    
                # Automatically detect if label is at the start or end
                try:
                    float(row[-1])
                    label = row[0]
                    features = [float(x) for x in row[1:]]
                except ValueError:
                    label = row[-1]
                    features = [float(x) for x in row[:-1]]
                
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
        # Determine input shape dynamically based on the dataset
        num_features = X.shape[1]
        model = Sequential([
            Dense(128, activation='relu', input_shape=(num_features,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(len(unique_labels), activation='softmax')
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


@app.route('/trainer')
@admin_required
def trainer():
    """Renders the Standalone Trainer UI inside the main app."""
    global camera_mode
    camera_mode = 'training_idle'
    return render_template('trainer.html', username=session.get('username', ''), is_admin=True)

@app.route('/start_recording', methods=['POST'])
@admin_required
def start_recording():
    global is_recording, current_label, frames_recorded, target_frames, camera_mode
    
    data = request.json
    label = data.get('label', '').strip().upper()
    frames = data.get('frames', 100)
    
    if not label:
        return jsonify({'success': False, 'error': 'Label is required'})
        
    current_label = label
    target_frames = int(frames)
    frames_recorded = 0
    is_recording = True
    camera_mode = 'training_idle'
    
    return jsonify({'success': True, 'message': f'Started recording for label: {label}'})

@app.route('/get_status')
@admin_required
def get_status():
    return jsonify({
        'is_recording': is_recording,
        'frames_recorded': frames_recorded,
        'target_frames': target_frames,
        'current_label': current_label
    })

@app.route('/toggle_testing', methods=['POST'])
@admin_required
def toggle_testing():
    global is_testing, custom_model, custom_labels, camera_mode
    
    data = request.json
    action = data.get('action')
    
    if action == 'start':
        if not load_custom_model():
            return jsonify({'success': False, 'error': 'No trained model found! Please train a model first.'})
        is_testing = True
        camera_mode = 'testing'
    else:
        is_testing = False
        camera_mode = 'training_idle'
        
    return jsonify({'success': True, 'is_testing': is_testing})

@app.route('/train_model', methods=['POST'])
@admin_required
def train_model():
    global is_training, training_log
    
    if is_training:
        return jsonify({'success': False, 'error': 'Training is already in progress'})
        
    is_training = True
    training_log = []
    
    # Start training in background thread
    thread = threading.Thread(target=train_model_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Training started in background'})

@app.route('/training_status')
@admin_required
def get_training_status():
    return jsonify({
        'is_training': is_training,
        'log': training_log
    })

@app.route('/delete_label', methods=['POST'])
@admin_required
def delete_label():
    """Removes all data for a specific label from the dataset CSV."""
    data = request.json
    label_to_delete = data.get('label', '').strip().upper()
    
    if not label_to_delete:
        return jsonify({'success': False, 'error': 'Label is required'})
        
    if not os.path.exists(csv_filepath):
        return jsonify({'success': False, 'error': 'Dataset file not found'})
        
    try:
        rows = []
        deleted_count = 0
        with open(csv_filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0:
                    # Check first or last column for label
                    if row[0].upper() == label_to_delete or row[-1].upper() == label_to_delete:
                        deleted_count += 1
                        continue
                    rows.append(row)
                    
        with open(csv_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            
        return jsonify({
            'success': True, 
            'message': f'Deleted {deleted_count} frames for label: {label_to_delete}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_dataset_labels')
@admin_required
def get_dataset_labels():
    """Fetches unique labels from the dataset CSV."""
    if not os.path.exists(csv_filepath):
        return jsonify({'success': True, 'labels': []})
        
    try:
        labels_found = set()
        with open(csv_filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 1:
                    # Skip header or malformed rows
                    if row[0].lower() == 'label' or not row[1].replace('.', '', 1).replace('-', '', 1).isdigit():
                        continue
                    
                    # Logic matches train_model_thread for label extraction
                    try:
                        float(row[-1])
                        labels_found.add(row[0].upper())
                    except ValueError:
                        labels_found.add(row[-1].upper())
                        
        sorted_labels = sorted(list(labels_found))
        return jsonify({'success': True, 'labels': sorted_labels})
    except Exception as e:
        print(f"Error fetching dataset labels: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin_dashboard():
    action = (request.form.get('action', '') or '').lower() if request.method == 'POST' else ''


    if request.method == 'POST' and action == 'add_user':
        new_username = request.form.get('new_username', '').strip()
        new_password = request.form.get('new_password', '')
        is_admin_flag = request.form.get('is_admin') == 'on'

        if not new_username or not new_password:
            flash('Username and Password required.', 'error')
            return redirect(url_for('admin_dashboard'))

        if len(new_username) < 3 or len(new_password) < 6:
            flash('Username (>3 chars) and Password (>6 chars) requirements not met.', 'error')
            return redirect(url_for('admin_dashboard'))

        if User.query.filter_by(username=new_username).first():
            flash('User already exists.', 'error')
            return redirect(url_for('admin_dashboard'))

        new_user = User(username=new_username, is_admin=is_admin_flag)
        new_user.set_password(new_password)
        db.session.add(new_user)
        db.session.commit()
        flash(f"User '{new_username}' added successfully.", 'success')
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST' and action == 'toggle_admin':
        user_id = request.form.get('user_id')
        user_to_toggle = User.query.get(user_id)
        if user_to_toggle:
            if user_to_toggle.username == 'Admin':
                flash('Cannot modify the primary Admin account.', 'error')
            elif user_to_toggle.id == session.get('user_id'):
                flash('Cannot demote your own account.', 'error')
            else:
                user_to_toggle.is_admin = not user_to_toggle.is_admin
                db.session.commit()
                flash(f"Updated admin status for '{user_to_toggle.username}'.", 'success')
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST' and action == 'delete_user':
        user_id = request.form.get('user_id')
        user_to_delete = User.query.get(user_id)
        if user_to_delete:
            if user_to_delete.username == 'Admin':
                flash('Cannot delete the primary Admin account.', 'error')
            elif user_to_delete.id == session.get('user_id'):
                flash('Cannot delete your own account.', 'error')
            else:
                db.session.delete(user_to_delete)
                db.session.commit()
                flash(f"User '{user_to_delete.username}' deleted.", 'success')
        return redirect(url_for('admin_dashboard'))

    all_users = User.query.order_by(User.id).all()
    pending_requests = WordRequest.query.filter_by(status='Pending', suggestion_type='ai_recognition').order_by(WordRequest.timestamp.desc()).all()
    
    return render_template(
        'admin_dashboard.html',
        username=session.get('username', ''),
        is_admin=True,
        users=all_users,
        pending_requests=pending_requests
    )

# ==========================================
# TEXT-TO-SIGN MODULE ENDPOINTS
# ==========================================

def load_sign_dictionary():
    try:
        with open(os.path.join(base_dir, 'sign_dictionary.json'), 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading sign dictionary: {e}")
        return {}

def save_sign_dictionary(data):
    try:
        with open(os.path.join(base_dir, 'sign_dictionary.json'), 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving sign dictionary: {e}")
        return False

def get_youtube_id(url):
    """Extract YouTube video ID from various URL formats."""
    if not url: return None
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

@app.route('/text_to_sign')
@login_required
def text_to_sign():
    user = User.query.get(session['user_id'])
    if not user:
        return redirect(url_for('login'))
    return render_template('text_to_sign.html', username=session.get('username', ''), is_admin=user.is_admin)

@app.route('/translate_text', methods=['POST'])
@login_required
def translate_text():
    data = request.json
    phrase = data.get('phrase', '').lower().strip()
    # Remove extra punctuation that could interfere with matching, keeping safe chars
    phrase = re.sub(r'[^a-z0-9\s]', '', phrase)
    
    dictionary = load_sign_dictionary()
    # Sort keys by length descending to match greedy multi-word phrases first
    keys = sorted(dictionary.keys(), key=len, reverse=True)
    
    results = []
    missing = []
    
    remaining = phrase
    while remaining:
        remaining = remaining.strip()
        if not remaining:
            break
            
        matched = False
        for k in keys:
            if remaining.startswith(k):
                # Ensure we match whole words, not partial prefixes
                if len(remaining) == len(k) or not remaining[len(k)].isalnum():
                    results.append({'word': k, 'youtube_id': dictionary[k]})
                    remaining = remaining[len(k):]
                    matched = True
                    break
                    
        if not matched:
            # Word not in dictionary, fallback to character-by-character
            word = remaining.split()[0]
            for char in word:
                if char in dictionary:
                    results.append({'word': char, 'youtube_id': dictionary[char]})
                elif char not in missing:
                    missing.append(char)
            if word not in missing:
                missing.append(word)
            remaining = remaining[len(word):]
            
    return jsonify({'results': results, 'missing': missing})

# --- Admin Dictionary Routes ---

@app.route('/admin/dictionary', methods=['GET'])
@admin_required
def admin_dictionary():
    dictionary = load_sign_dictionary()
    pending_requests = WordRequest.query.filter_by(status='Pending', suggestion_type='text_to_sign').order_by(WordRequest.timestamp.desc()).all()
    return render_template('admin_dictionary.html', username=session.get('username', ''), is_admin=True, dictionary=dictionary, pending_requests=pending_requests)

@app.route('/admin/dictionary/add', methods=['POST'])
@admin_required
def admin_dictionary_add():
    data = request.json
    phrase = data.get('phrase', '').strip().lower() # Normalization
    url = data.get('url', '').strip()
    
    if not phrase or not url:
        return jsonify({'success': False, 'error': 'Phrase and Link are required.'})
        
    yt_id = get_youtube_id(url)
    if not yt_id:
        # Fallback in case they pasted the ID directly
        if len(url) == 11 and url.isalnum() or '-' in url or '_' in url:
            yt_id = url
        else:
            return jsonify({'success': False, 'error': 'Invalid YouTube Link or ID.'})
        
    dictionary = load_sign_dictionary()
    dictionary[phrase] = yt_id
    
    if save_sign_dictionary(dictionary):
        return jsonify({'success': True, 'message': 'Word added to dictionary successfully!'})
    return jsonify({'success': False, 'error': 'Failed to save dictionary to file.'})

@app.route('/admin/dictionary/bulk_add', methods=['POST'])
@admin_required
def admin_dictionary_bulk_add():
    data = request.json
    entries = data.get('entries', [])
    
    if not entries:
        return jsonify({'success': False, 'error': 'No entries provided.'})
        
    dictionary = load_sign_dictionary()
    added_count = 0
    errors = []
    
    for entry in entries:
        phrase = entry.get('phrase', '').strip().lower()
        url = entry.get('url', '').strip()
        
        if not phrase or not url:
            continue
            
        yt_id = get_youtube_id(url)
        if not yt_id:
             if len(url) == 11 and url.isalnum() or '-' in url or '_' in url:
                 yt_id = url
             else:
                 errors.append(f"Invalid URL for '{phrase}'")
                 continue
                 
        dictionary[phrase] = yt_id
        added_count += 1
        
    if added_count > 0:
        if save_sign_dictionary(dictionary):
            return jsonify({'success': True, 'message': f'Successfully imported {added_count} words.', 'errors': errors})
        return jsonify({'success': False, 'error': 'Failed to save dictionary to file.'})
    return jsonify({'success': False, 'error': 'No valid entries to add.', 'errors': errors})

@app.route('/admin/dictionary/edit', methods=['POST'])
@admin_required
def admin_dictionary_edit():
    data = request.json
    old_phrase = data.get('old_phrase', '').strip()
    new_phrase = data.get('new_phrase', '').strip().lower()
    new_url = data.get('new_url', '').strip()
    
    if not old_phrase or not new_phrase or not new_url:
        return jsonify({'success': False, 'error': 'All fields are required.'})
        
    yt_id = get_youtube_id(new_url)
    if not yt_id:
        if len(new_url) == 11 and new_url.isalnum() or '-' in new_url or '_' in new_url:
            yt_id = new_url
        else:
            return jsonify({'success': False, 'error': 'Invalid YouTube Link or ID.'})
            
    dictionary = load_sign_dictionary()
    
    if old_phrase in dictionary:
        if old_phrase != new_phrase:
            del dictionary[old_phrase]
        dictionary[new_phrase] = yt_id
        
        if save_sign_dictionary(dictionary):
            return jsonify({'success': True, 'message': 'Entry updated successfully!'})
        return jsonify({'success': False, 'error': 'Failed to save dictionary to file.'})
        
    return jsonify({'success': False, 'error': 'Original phrase not found.'})

@app.route('/admin/dictionary/delete', methods=['POST'])
@admin_required
def admin_dictionary_delete():
    data = request.json
    phrase = data.get('phrase', '').strip()
    
    if not phrase:
        return jsonify({'success': False, 'error': 'Phrase is required.'})
        
    dictionary = load_sign_dictionary()
    if phrase in dictionary:
        del dictionary[phrase]
        if save_sign_dictionary(dictionary):
            return jsonify({'success': True, 'message': 'Entry deleted globally.'})
        return jsonify({'success': False, 'error': 'Failed to save dictionary to file.'})
        
    return jsonify({'success': False, 'error': 'Phrase not found.'})

# ==========================================
# FEEDBACK / REQUEST MODULE ENDPOINTS
# ==========================================

@app.route('/submit_word_request', methods=['POST'])
@login_required
def submit_word_request():
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'error': 'User not found'})
        
    data = request.json
    word = data.get('word', '').strip()
    
    if not word:
        return jsonify({'success': False, 'error': 'Word cannot be empty'})
        
    if len(word) > 50:
        return jsonify({'success': False, 'error': 'Word is too long'})
        
    # Check if this user recently requested this exact word
    existing = WordRequest.query.filter_by(user_id=user.id, requested_word=word, status='Pending').first()
    if existing:
        return jsonify({'success': False, 'error': 'You already have a pending request for this word.'})
        
    suggestion_type = data.get('type', 'ai_recognition')
        
    new_request = WordRequest(user_id=user.id, requested_word=word, suggestion_type=suggestion_type)
    db.session.add(new_request)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Request submitted successfully!'})

@app.route('/update_word_request', methods=['POST'])
@admin_required
def update_word_request():
    data = request.json
    req_id = data.get('request_id')
    new_status = data.get('status')
    
    if new_status not in ['Added', 'Rejected']:
        return jsonify({'success': False, 'error': 'Invalid status'})
        
    word_req = WordRequest.query.get(req_id)
    if not word_req:
        return jsonify({'success': False, 'error': 'Request not found'})
        
    word_req.status = new_status
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/my_requests', methods=['GET'])
@login_required
def my_requests():
    # Only fetches the current user's requests for the dashboard widget
    user_requests = WordRequest.query.filter_by(user_id=session['user_id']).order_by(WordRequest.timestamp.desc()).all()
    requests_data = [{'word': r.requested_word, 'status': r.status, 'date': r.timestamp.strftime('%Y-%m-%d')} for r in user_requests]
    return jsonify({'requests': requests_data})

@app.route('/recently_added', methods=['GET'])
@login_required
def recently_added():
    # Fetches recently approved requests to show community progress
    recent = WordRequest.query.filter_by(status='Added').order_by(WordRequest.timestamp.desc()).limit(5).all()
    recent_data = [r.requested_word for r in recent]
    # Unique filter while preserving order
    seen = set()
    unique_recent = [x for x in recent_data if not (x in seen or seen.add(x))]
    return jsonify({'recent': unique_recent})

# Load default model natively on startup after all globals have initialized
load_model(active_mode)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)