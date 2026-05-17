# 🖐️ Hand Sign Detection & Translation System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-00C0FF?style=flat&logo=google&logoColor=white)](https://mediapipe.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

An advanced, real-time sign language recognition and educational platform. This system bridges the communication gap between the hearing-impaired community and the general public using cutting-edge computer vision and deep learning.

---

## 🚀 Features Overview

| Feature | Description | Icon |
| :--- | :--- | :---: |
| **Real-time Recognition** | Dual-hand tracking for static and dynamic gestures. | ⚡ |
| **Interactive Learning** | Gamified tutorials and quizzes with live feedback. | 🎓 |
| **Text-to-Sign** | Visual dictionary mapping text to sign videos. | 📖 |
| **Advanced Analytics** | Data-driven insights into learning progress. | 📊 |
| **Admin Suite** | On-the-fly model training and system management. | 🛠️ |

---

## 📦 Core Modules

### 1. 🔐 Authentication System
*   **Secure Access**: Robust login and registration system with password hashing.
*   **Identity Recovery**: Multi-step password reset using customizable security questions.
*   **Security Layers**: Integrated CSRF protection, rate limiting, and secure HTTP headers via Talisman.

### 2. 👤 Manage Account
*   **Profile Customization**: Users can update their credentials and security settings.
*   **Progress Tracking**: Integrated with the tutorial system to track completion status.

### 3. ⚙️ Manage Settings
*   **Hardware Control**: Intelligent camera discovery using DirectShow.
*   **User Preference**: Persistent storage of camera indices and feed preferences.

### 4. 🎓 Manage Learning Module
*   **Tutorial Levels**: 
    *   **Basics**: Focused on the manual alphabet (A-Z).
    *   **Intermediate**: Common words (e.g., "Makan", "Sayang").
    *   **Advanced**: Full conversational phrases.
*   **Quiz System**: Real-time assessment where users must perform the correct sign to advance.

### 5. 👐 Sign-to-Text Translation (Live Recognition)
*   **Hybrid Model**: Combines MediaPipe hand landmarks with a custom MLP (Multi-Layer Perceptron) architecture.
*   **Gesture Support**: High-accuracy detection for both static hand shapes and dynamic movement patterns.

### 6. 📝 Text-to-Sign Translation
*   **Visual Dictionary**: Maps input text to high-quality sign language video demonstrations.
*   **Smart Fallback**: Automatically reverts to character-by-character spelling for unrecognized words.
*   **Greedy Matching**: Prioritizes multi-word phrases over individual words for natural translation.

### 7. 📈 Generate Analytical Data
*   **User Statistics**: Visualizes practice frequency and top performing users.
*   **Sign Popularity**: Tracks which signs are most frequently practiced or requested.
*   **System Health**: Real-time monitoring of CPU usage, memory consumption, and model latency.

---

## 🛠️ Admin Guide

The system includes a powerful Administrative Dashboard for system maintainers:

### 👤 User Management
*   Full CRUD (Create, Read, Update, Delete) operations for user accounts.
*   Grant or revoke Administrative privileges with a single click.

### 🧠 Model Trainer & Updater
*   **Data Collection**: Record new hand signs directly through the browser (standardized 100 frames/sample).
*   **Dynamic Training**: Trigger background training threads for both Static and Dynamic gesture models.
*   **Instant Update**: Deploy newly trained models (`.h5`) to the production environment without restarting the server.

### 📖 Dictionary Manager
*   Add new vocabulary by mapping phrases to YouTube video IDs.
*   Bulk import/export dictionary data via CSV.
*   Manage user-submitted word requests.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
*   Python 3.8 or higher
*   A webcam (for recognition features)

### 2. Clone and Install
```bash
git clone https://github.com/Moriluna/Hand-Sign-Detection2.git
cd Hand-Sign-Detection2
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
SECRET_KEY=your_secure_random_key
DEBUG=True
```

### 4. Database Initialization
The system uses SQLite (via SQLAlchemy). The database schema is automatically created upon the first run.
```bash
python app2.py
```
*Default Admin Credentials:* `Admin` / `admin123`

---

## 🏗️ Project Architecture

The system operates on a sophisticated pipeline to ensure low-latency recognition:

1.  **Frame Capture**: OpenCV captures raw video input.
2.  **Preprocessing**: MediaPipe extracts 21 3D-coordinates (landmarks) per hand.
3.  **Feature Extraction**: Coordinates are normalized relative to the wrist and scaled.
4.  **Inference**:
    *   **Static**: Landmark data is fed into an MLP model.
    *   **Dynamic**: A temporal buffer (16 frames) is analyzed for motion patterns.
5.  **UI Feedback**: Results are piped to the frontend via Flask-SocketIO or AJAX polling for immediate user response.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
*Developed with ❤️ for the Hearing-Impaired Community.*
