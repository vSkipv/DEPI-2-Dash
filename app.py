from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
from datetime import datetime, timedelta
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from model import SkinCancerModel

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skin_cancer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize model
model = SkinCancerModel()
if not model.load_model():
    print("Creating new model...")
    model.create_model()
    model.save_model()

db = SQLAlchemy(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    scans = db.relationship('Scan', backref='user', lazy=True)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Get statistics for dashboard
    total_scans = Scan.query.count()
    high_risk = Scan.query.filter(Scan.prediction == 'Malignant').count()
    benign = Scan.query.filter(Scan.prediction == 'Benign').count()
    
    # Get recent scans
    recent_scans = Scan.query.order_by(Scan.date.desc()).limit(10).all()
    
    return render_template('dashboard.html', 
                         total_scans=total_scans,
                         high_risk=high_risk,
                         benign=benign,
                         recent_scans=recent_scans)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction from model
            prediction, confidence = model.predict(filepath)
            
            if prediction is None:
                return jsonify({'error': 'Error processing image'}), 500
            
            # Save to database
            new_scan = Scan(
                image_path=filepath,
                prediction=prediction,
                confidence=confidence,
                user_id=1  # Replace with actual user ID from session
            )
            db.session.add(new_scan)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'confidence': confidence,
                'image_path': filepath
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        


@app.route('/api/stats')
def get_stats():
    """API endpoint for dashboard statistics"""
    total_scans = Scan.query.count()
    bcc_cases = Scan.query.filter(Scan.prediction == 'BCC').count()
    ack_cases = Scan.query.filter(Scan.prediction == 'ACK').count()
    scc_cases = Scan.query.filter(Scan.prediction == 'SCC').count()
    nev_cases = Scan.query.filter(Scan.prediction == 'NEV').count()
    mel_cases = Scan.query.filter(Scan.prediction == 'MEL').count()
    sek_cases = Scan.query.filter(Scan.prediction == 'SEK').count()

    # Calculate distribution for the chart
    distribution = [
        bcc_cases,
        ack_cases,
        scc_cases,
        nev_cases,
        mel_cases,
        sek_cases
    ]

    # Get weekly trends
    weekly_trends = []
    for i in range(4):
        week_start = datetime.utcnow() - timedelta(days=(3-i)*7)
        week_end = datetime.utcnow() - timedelta(days=(2-i)*7)
        count = Scan.query.filter(
            Scan.date >= week_start,
            Scan.date < week_end
        ).count()
        weekly_trends.append(count)
    
    # For the new card: scans in last 30 days
    last_30_days = datetime.utcnow() - timedelta(days=30)
    last30DaysScans = Scan.query.filter(Scan.date >= last_30_days).count()

    return jsonify({
        'totalScans': total_scans,
        'bccCases': bcc_cases,
        'ackCases': ack_cases,
        'sccCases': scc_cases,
        'nevCases': nev_cases,
        'melCases': mel_cases,
        'sekCases': sek_cases,
        'distribution': distribution,
        'trends': weekly_trends,
        'last30DaysScans': last30DaysScans
    })

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 