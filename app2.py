from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, flash
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import shutil
from datetime import datetime
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

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    camera_enabled = db.Column(db.Boolean, default=True)
    camera_index = db.Column(db.Integer, default=0)
    is_admin = db.Column(db.Boolean, default=False)
    tutorial_completed = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

with app.app_context():
    db.create_all()
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

# Model Paths
spelling_model_path = os.path.join(base_dir, "Model", "AtoZ.h5")
spelling_labels_path = os.path.join(base_dir, "Model", "labelsAtoZ.txt")
words_model_path = os.path.join(base_dir, "Model", "dynamic_model.h5")
words_labels_path = os.path.join(base_dir, "Model", "dynamic_labels.txt")

detector = HandDetector(maxHands=1, detectionCon=0.8)
classifier = None
labels = []

def load_model(mode):
    global classifier, labels, active_mode
    active_mode = mode
    
    if mode == "spelling":
        model_p = spelling_model_path
        labels_p = spelling_labels_path
    elif mode == "words":
        model_p = words_model_path
        labels_p = words_labels_path
    else:
        return False
        
    try:
        # Load labels
        with open(labels_p, 'r') as f:
            lines = f.readlines()
            if mode == "spelling":
                # Spelling labels have format "0 A"
                labels = [line.strip().split(" ")[1] if " " in line else line.strip() for line in lines if line.strip()]
            else:
                # Word labels are just plain text lines
                labels = [line.strip() for line in lines if line.strip()]
                
        # Load classification model
        classifier = Classifier(model_p, labels_p)
        print(f"✅ Successfully loaded {mode} model")
        return True
    except Exception as e:
        print(f"❌ Error loading {mode} model: {str(e)}")
        # Fallback to simple generic labels if model missing
        labels = ["A", "B", "C", "D"] if mode == "spelling" else ["Word1", "Word2"]
        classifier = None
        return False

# Load default model on startup
load_model(active_mode)

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


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def generate_frames():
    global current_prediction, confidence_scores, camera_active, current_camera_index

    cap = None

    while True:
        # Check if camera should be active and reinitialize if needed
        if camera_active and (cap is None or not cap.isOpened()):
            cap = cv2.VideoCapture(current_camera_index)
            if not cap.isOpened():
                # Return a black frame with error message
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "Camera not available", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue

        if not camera_active:
            # Return a black frame with "Camera Disabled" message
            if cap is not None:
                cap.release()
                cap = None

            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "Camera Disabled", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', black_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        success, img = cap.read()
        if not success:
            print("⚠️ Camera frame not captured!")
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Safety crop check with padding instead of cutting off
            # This ensures the hand isn't distorted when it's near the edge of the frame
            y1 = y - offset
            y2 = y + h + offset
            x1 = x - offset
            x2 = x + w + offset

            # Calculate padding needed if box goes outside image
            pad_top = max(0, -y1)
            pad_bottom = max(0, y2 - img.shape[0])
            pad_left = max(0, -x1)
            pad_right = max(0, x2 - img.shape[1])

            # Apply cropping (constrained to image boundaries)
            y1_safe = max(0, y1)
            y2_safe = min(img.shape[0], y2)
            x1_safe = max(0, x1)
            x2_safe = min(img.shape[1], x2)
            
            imgCrop = img[y1_safe:y2_safe, x1_safe:x2_safe]

            if imgCrop.size == 0:
                continue
                
            # Pad the cropped image back to the expected size so aspect ratio is preserved
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                imgCrop = cv2.copyMakeBorder(imgCrop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
                
            aspectRatio = imgCrop.shape[0] / imgCrop.shape[1] if imgCrop.shape[1] > 0 else 1
            prediction = None
            index = 0

            # More robust resizing and centering logic
            if aspectRatio > 1:
                k = imgSize / imgCrop.shape[0]
                wCal = math.ceil(k * imgCrop.shape[1])

                if wCal > 0:
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    # Safe placement into imgWhite
                    place_width = min(imgResize.shape[1], imgSize - wGap)
                    imgWhite[:, wGap:wGap + place_width] = imgResize[:, :place_width]
            else:
                k = imgSize / imgCrop.shape[1]
                hCal = math.ceil(k * imgCrop.shape[0])

                if hCal > 0:
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    # Safe placement into imgWhite
                    place_height = min(imgResize.shape[0], imgSize - hGap)
                    imgWhite[hGap:hGap + place_height, :] = imgResize[:place_height, :]
            
            try:
                # Make prediction if classifier is loaded
                if classifier:
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    
                    if 0 <= index < len(labels):
                        current_prediction = labels[index]
                        
                        # Store simulated confidence (Classifier in cvzone doesn't return raw prob array by default)
                        # So we generate a mock confidence array with the dominant class having high confidence
                        mock_conf = [0.0] * len(labels)
                        mock_conf[index] = 0.95 
                        confidence_scores = mock_conf
                        
                        # Add aesthetic UI drawing on frame
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, current_prediction, (x, y - 26),
                                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset),
                                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
                else:
                    cv2.putText(imgOutput, "Model not loaded", (x, y - 26),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error during prediction: {e}")
                current_prediction = "Prediction Error"
                confidence_scores = []
        else:
            current_prediction = "No hand detected"
            confidence_scores = []

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


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

        # Validation
        if not username or not password:
            flash('Username and password are required', 'error')
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
        new_user = User(username=username)
        new_user.set_password(password)

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
    global camera_active, current_camera_index
    user = User.query.get(session['user_id'])

    if user is None:
        session.clear()
        flash('Session expired. Please login again.', 'error')
        return redirect(url_for('login'))

    if not user.tutorial_completed:
        return redirect(url_for('tutorial'))

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

        user.username = new_username
        session['username'] = new_username

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

    # Get available cameras
    # Get available cameras safely (Prevent Windows crashes)
    available_cameras = []
    for i in range(2):  # Only check the first 2 indices to prevent hanging
        # Use CAP_DSHOW for faster, safer checks on Windows
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    return render_template('settings.html',
                           user=user,
                           available_cameras=available_cameras)


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
    user = db.session.get(User, session['user_id'])
    if user is None:
        session.clear()
        return redirect(url_for('login'))
    return render_template('quiz.html', username=session.get('username', ''), is_admin=user.is_admin)


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

MODEL_BACKUP_DIR = os.path.join(base_dir, "Model", "Backup Model")
ALLOWED_MODEL_EXT = {"h5"}
ALLOWED_LABEL_EXT = {"txt"}

def has_allowed_extension(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext

def backup_existing_model_files():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(MODEL_BACKUP_DIR, timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    if os.path.exists(spelling_model_path):
        shutil.copy2(spelling_model_path, os.path.join(backup_dir, os.path.basename(spelling_model_path)))
    if os.path.exists(spelling_labels_path):
        shutil.copy2(spelling_labels_path, os.path.join(backup_dir, os.path.basename(spelling_labels_path)))
    return backup_dir

def list_model_backups():
    backups = []
    if os.path.isdir(MODEL_BACKUP_DIR):
        for name in sorted(os.listdir(MODEL_BACKUP_DIR), reverse=True):
            b_path = os.path.join(MODEL_BACKUP_DIR, name)
            if os.path.isdir(b_path):
                backups.append({"name": name})
    return backups

def restore_model_from_backup(backup_name):
    backup_dir = os.path.join(MODEL_BACKUP_DIR, backup_name)
    m_path = os.path.join(backup_dir, os.path.basename(spelling_model_path))
    l_path = os.path.join(backup_dir, os.path.basename(spelling_labels_path))
    if os.path.exists(m_path):
        shutil.copy2(m_path, spelling_model_path)
    if os.path.exists(l_path):
        shutil.copy2(l_path, spelling_labels_path)

def reload_model_from_disk():
    # Only reload if the active mode is spelling
    if active_mode == "spelling":
        load_model("spelling")

@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin_dashboard():
    action = (request.form.get('action', '') or '').lower() if request.method == 'POST' else ''

    if action == 'rollback':
        backup_name = (request.form.get('backup_name', '') or '').strip()
        if not backup_name:
            flash('Choose a backup to restore.', 'error')
            return redirect(url_for('admin_dashboard'))
        try:
            backup_existing_model_files()
            restore_model_from_backup(backup_name)
            reload_model_from_disk()
            flash('Rollback completed successfully.', 'success')
        except Exception as exc:
            flash(f'Rollback failed: {exc}', 'error')
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST' and action == 'update':
        model_file = request.files.get('model_file')
        labels_file = request.files.get('labels_file')

        if not model_file or not model_file.filename:
            flash('Upload a .h5 model file.', 'error')
            return redirect(url_for('admin_dashboard'))
        if not labels_file or not labels_file.filename:
            flash('Upload a .txt labels file.', 'error')
            return redirect(url_for('admin_dashboard'))
        if not has_allowed_extension(model_file.filename, ALLOWED_MODEL_EXT):
            flash('Model file must be .h5', 'error')
            return redirect(url_for('admin_dashboard'))
        if not has_allowed_extension(labels_file.filename, ALLOWED_LABEL_EXT):
            flash('Labels file must be .txt', 'error')
            return redirect(url_for('admin_dashboard'))

        try:
            backup_existing_model_files()
            model_file.save(spelling_model_path)
            labels_file.save(spelling_labels_path)
            reload_model_from_disk()
            flash('Model and labels updated successfully.', 'success')
        except Exception as exc:
            flash(f'Failed to update model: {exc}', 'error')
        return redirect(url_for('admin_dashboard'))

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

    backups = list_model_backups()
    all_users = User.query.order_by(User.id).all()
    return render_template(
        'admin_dashboard.html',
        username=session.get('username', ''),
        is_admin=True,
        spelling_backups=backups,
        users=all_users
    )

if __name__ == '__main__':
    app.run(debug=True, threaded=True)