"""
FreshGuard AI - Intelligent Food Freshness Detection and Classification System
Main Flask Application
Models: Support Vector Machine (SVM) and K-Nearest Neighbors (KNN)
Student Name: Hariom Gupta
UID: 25MCA20022
Strict Rule: Exclusively uses MySQL 8.0 (No SQLite allowed).
"""

import os
import sys
import json
import uuid
import joblib
import numpy as np
from datetime import datetime
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import (
    Flask, render_template, request, redirect, 
    url_for, flash, jsonify, send_from_directory, abort
)
from werkzeug.utils import secure_filename

from config import Config
from database.db_connection import check_db_health, init_db
from database.prediction_repository import PredictionRepository
from training.preprocess import (
    extract_hog_and_color_features, 
    parse_class_result,
    CLASS_NAMES, 
    IDX_TO_CLASS
)

# Initialize Flask
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# Ensure uploads directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Global model cache
MODELS_CACHE = {
    'svm': None,
    'knn': None,
    'metadata': None
}

def get_svm_artifact():
    if MODELS_CACHE['svm'] is None:
        if os.path.exists(Config.SVM_MODEL_PATH):
            print("[FreshGuard] Loading SVM model from disk...")
            MODELS_CACHE['svm'] = joblib.load(Config.SVM_MODEL_PATH)
        else:
            print("[FreshGuard Warning] SVM model file not found.")
    return MODELS_CACHE['svm']

def get_knn_artifact():
    if MODELS_CACHE['knn'] is None:
        if os.path.exists(Config.KNN_MODEL_PATH):
            print("[FreshGuard] Loading KNN model from disk...")
            MODELS_CACHE['knn'] = joblib.load(Config.KNN_MODEL_PATH)
        else:
            print("[FreshGuard Warning] KNN model file not found.")
    return MODELS_CACHE['knn']

def get_metadata():
    if MODELS_CACHE['metadata'] is None:
        if os.path.exists(Config.METADATA_PATH):
            with open(Config.METADATA_PATH, 'r') as f:
                MODELS_CACHE['metadata'] = json.load(f)
    return MODELS_CACHE['metadata']

# Initialize repository
repo = PredictionRepository()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# ----------------- ROUTES -----------------

@app.route('/')
def index():
    """Home landing page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard showing MySQL stats and model summaries"""
    try:
        stats = repo.get_dashboard_stats()
    except Exception as e:
        flash(f"Database error: {e}", "danger")
        stats = {
            'total_predictions': 0,
            'total_fresh': 0,
            'total_rotten': 0,
            'total_uncertain': 0,
            'avg_confidence': 0.0,
            'model_breakdown': [],
            'top_categories': [],
            'recent_predictions': []
        }
    metadata = get_metadata()
    return render_template('dashboard.html', stats=stats, metadata=metadata)

import shutil

@app.route('/upload')
def upload_page():
    """Image upload interface with verification samples"""
    samples = [
        {'id': 'fresh_apple', 'name': 'Fresh Apple', 'type': 'Apple', 'status': 'FRESH', 'img': 'images/samples/fresh_apple.png'},
        {'id': 'rotten_apple', 'name': 'Rotten Apple', 'type': 'Apple', 'status': 'ROTTEN', 'img': 'images/samples/rotten_apple.png'},
        {'id': 'fresh_banana', 'name': 'Fresh Banana', 'type': 'Banana', 'status': 'FRESH', 'img': 'images/samples/fresh_banana.png'},
        {'id': 'rotten_banana', 'name': 'Rotten Banana', 'type': 'Banana', 'status': 'ROTTEN', 'img': 'images/samples/rotten_banana.png'},
        {'id': 'fresh_orange', 'name': 'Fresh Orange', 'type': 'Orange', 'status': 'FRESH', 'img': 'images/samples/fresh_orange.png'},
        {'id': 'rotten_orange', 'name': 'Rotten Orange', 'type': 'Orange', 'status': 'ROTTEN', 'img': 'images/samples/rotten_orange.png'}
    ]
    return render_template('upload.html', samples=samples)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload or sample selection, runs inference (SVM or KNN), saves record to MySQL, and redirects to result"""
    sample_name = request.form.get('sample_name', '').strip()

    if sample_name:
        # User selected an instant verification sample
        sample_file = os.path.join(BASE_DIR, 'static', 'images', 'samples', f"{sample_name}.png")
        if not os.path.exists(sample_file):
            flash(f"Verification sample image '{sample_name}' not found.", "error")
            return redirect(url_for('upload_page'))
        unique_filename = f"sample_{sample_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
        saved_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        shutil.copy2(sample_file, saved_path)
    else:
        # User uploaded a custom image file
        if 'image' not in request.files:
            flash("No image file provided in upload request.", "error")
            return redirect(url_for('upload_page'))
        
        file = request.files['image']
        if file.filename == '':
            flash("No image selected. Please upload a file or choose one of the verification samples below.", "error")
            return redirect(url_for('upload_page'))
            
        if not allowed_file(file.filename):
            flash("Invalid file extension. Please upload a JPG, JPEG, or PNG image.", "error")
            return redirect(url_for('upload_page'))

        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{ext}"
        saved_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        file.save(saved_path)

    model_choice = request.form.get('model_choice', 'SVM').upper()
    if model_choice not in ['SVM', 'KNN']:
        model_choice = 'SVM'

    # Validate image file integrity
    try:
        with Image.open(saved_path) as img:
            img.verify()
    except Exception as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        flash(f"Selected file is not a valid image: {e}", "error")
        return redirect(url_for('upload_page'))

    # Run Inference
    try:
        features = extract_hog_and_color_features(saved_path)

        if model_choice == 'SVM':
            svm_artifact = get_svm_artifact()
            if svm_artifact is None:
                raise RuntimeError("SVM model is not available. Please run training first.")
            features_scaled = svm_artifact['scaler'].transform([features])
            probs = svm_artifact['model'].predict_proba(features_scaled)[0]
            pred_idx = int(np.argmax(probs))
            raw_label = IDX_TO_CLASS[pred_idx]
            confidence = float(probs[pred_idx])

        elif model_choice == 'KNN':
            knn_artifact = get_knn_artifact()
            if knn_artifact is None:
                raise RuntimeError("KNN model is not available. Please run training first.")
            features_scaled = knn_artifact['scaler'].transform([features])
            probs = knn_artifact['model'].predict_proba(features_scaled)[0]
            pred_idx = int(np.argmax(probs))
            raw_label = IDX_TO_CLASS[pred_idx]
            confidence = float(probs[pred_idx])

        # Parse freshness and category
        parsed = parse_class_result(raw_label, confidence, threshold=Config.CONFIDENCE_THRESHOLD)

        # Strictly insert into MySQL
        inserted_id = repo.insert_prediction(
            image_filename=unique_filename,
            predicted_category=parsed['predicted_category'],
            freshness_status=parsed['freshness_status'],
            confidence_score=parsed['confidence_score'],
            model_name=model_choice
        )

        return redirect(url_for('result_page', prediction_id=inserted_id))

    except Exception as err:
        flash(f"Prediction failed: {err}", "danger")
        return redirect(url_for('upload_page'))

@app.route('/result/<int:prediction_id>')
def result_page(prediction_id):
    """Displays detailed classification results for a prediction stored in MySQL"""
    record = repo.get_prediction_by_id(prediction_id)
    if not record:
        flash(f"Prediction record #{prediction_id} not found in MySQL.", "warning")
        return redirect(url_for('dashboard'))

    fake_raw = f"{record['freshness_status'].lower()}_{record['predicted_category'].lower()}"
    parsed = parse_class_result(fake_raw, record['confidence_score'], threshold=Config.CONFIDENCE_THRESHOLD)
    
    return render_template('result.html', record=record, explanation=parsed['explanation'])

@app.route('/comparison')
def comparison():
    """Model comparison dashboard showing real empirical evaluation metrics and plots (SVM vs KNN)"""
    metadata = get_metadata()
    report_text = ""
    report_path = os.path.join(Config.REPORTS_DIR, 'classification_report.txt')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_text = f.read()

    return render_template(
        'comparison.html', 
        metadata=metadata, 
        classification_report_text=report_text
    )

@app.route('/history')
def history():
    """Prediction history page with MySQL search, filtering, and pagination"""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '', type=str)
    status = request.args.get('status', '', type=str)
    model = request.args.get('model', '', type=str)
    per_page = 12

    try:
        records, total_count = repo.get_all_predictions(
            page=page,
            per_page=per_page,
            category_search=search,
            status_filter=status,
            model_filter=model
        )
    except Exception as e:
        flash(f"Failed to fetch records from MySQL: {e}", "danger")
        records, total_count = [], 0

    total_pages = max(1, (total_count + per_page - 1) // per_page)

    return render_template(
        'history.html',
        records=records,
        total_records=total_count,
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        category_search=search,
        status_filter=status,
        model_filter=model
    )

@app.route('/history/delete/<int:prediction_id>', methods=['POST'])
def delete_history(prediction_id):
    """Deletes a specific record from MySQL"""
    try:
        record = repo.get_prediction_by_id(prediction_id)
        if record:
            local_path = os.path.join(Config.UPLOAD_FOLDER, record['image_filename'])
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
            repo.delete_prediction(prediction_id)
            flash(f"Record #{prediction_id} successfully deleted from MySQL.", "success")
        else:
            flash(f"Record #{prediction_id} not found.", "warning")
    except Exception as e:
        flash(f"Deletion failed: {e}", "danger")
    return redirect(url_for('history'))

@app.route('/history/clear', methods=['POST'])
def clear_history_route():
    """Truncates the prediction_history table in MySQL"""
    try:
        repo.clear_history()
        flash("All prediction history successfully cleared from MySQL database.", "success")
    except Exception as e:
        flash(f"Clear history failed: {e}", "danger")
    return redirect(url_for('history'))

@app.route('/about')
def about():
    """About project information page"""
    return render_template('about.html')

@app.route('/reports/<path:filename>')
def get_report_image(filename):
    """Serves report graphs and confusion matrix heatmaps"""
    return send_from_directory(Config.REPORTS_DIR, filename)

@app.route('/health')
def health():
    """System health API endpoint"""
    db_ok, db_msg = check_db_health()
    return jsonify({
        'status': 'healthy' if db_ok else 'degraded',
        'mysql_database': {
            'connected': db_ok,
            'message': db_msg,
            'database_name': Config.DB_NAME,
            'port': Config.DB_PORT
        },
        'models': {
            'svm_loaded': MODELS_CACHE['svm'] is not None or os.path.exists(Config.SVM_MODEL_PATH),
            'knn_loaded': MODELS_CACHE['knn'] is not None or os.path.exists(Config.KNN_MODEL_PATH)
        },
        'student': {
            'name': Config.STUDENT_NAME,
            'uid': Config.STUDENT_UID,
            'project': Config.PROJECT_NAME
        }
    })

# ----------------- INITIALIZATION -----------------

if __name__ == '__main__':
    print("=" * 60)
    print("Starting FreshGuard AI Application Server (SVM & KNN)")
    print(f"Student: {Config.STUDENT_NAME} ({Config.STUDENT_UID})")
    print("Strict Database: MySQL 8.0 (SQLite strictly prohibited)")
    print("=" * 60)

    # Initialize MySQL schema
    init_db()

    # Preload models into cache
    get_svm_artifact()
    get_knn_artifact()
    get_metadata()

    # Start Flask development server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
