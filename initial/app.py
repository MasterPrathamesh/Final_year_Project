from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///eye.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)

class Diagnosis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_phone = db.Column(db.String(20), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    diagnosis_result = db.Column(db.String(200))
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Handle form submission
        patient_name = request.form.get('name')
        patient_phone = request.form.get('phone')
        file = request.files.get('eye_image')

        # Validation
        if not all([patient_name, patient_phone, file]):
            flash('All fields are required!', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Create new diagnosis record
            new_diagnosis = Diagnosis(
                patient_name=patient_name,
                patient_phone=patient_phone,
                image_path=file_path,
                diagnosis_result="Pending Analysis"  # Replace with actual diagnosis
            )
            
            try:
                db.session.add(new_diagnosis)
                db.session.commit()
                flash('Image uploaded successfully! Analysis in progress.', 'success')
                return redirect(url_for('upload'))
            except Exception as e:
                db.session.rollback()
                flash('Error saving to database!', 'danger')
        
        else:
            flash('Allowed image types are: png, jpg, jpeg', 'danger')

    return render_template('upload.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create upload directory if not exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)