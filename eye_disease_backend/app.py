from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

# Import the model instances
from swin_model import swin_model
from convnext_model import convnext_model
from coatnet_model import coatnet_model
from efficientformer_model import efficientformer_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'Backend is running successfully!'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Perform predictions
                result = {
                    'convnext': {
                        'prediction': convnext_model.predict(filepath)
                    },
                    'swin': {
                        'prediction': swin_model.predict(filepath)
                    },
                    'coatnet': {
                        'prediction': coatnet_model.predict(filepath)
                    },
                    'efficientformer': {
                        'prediction': efficientformer_model.predict(filepath)
                    }
                }

                # Format results
                formatted_result = {}
                for model_name in result:
                    class_name, confidence = result[model_name]['prediction']
                    formatted_result[model_name] = {
                        'predicted_class': class_name,
                        'confidence': float(confidence)
                    }

                return jsonify(formatted_result), 200

            except Exception as e:
                return jsonify({'error': f'Prediction error: {str(e)}'}), 500

            finally:
                # Clean up
                if os.path.exists(filepath):
                    os.remove(filepath)

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)