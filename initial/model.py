# model.py
from flask import Flask, request, jsonify
from PIL import Image
import io
import os  # Added for path handling

app = Flask(__name__)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from transformers import AutoModelForImageClassification, AutoImageProcessor
from safetensors.torch import load_file
import torch.nn.functional as F

# Load model architecture based on the model name
def load_model_architecture(model_name):
    if "ConvNeXt" in model_name:
        model = timm.create_model('convnext_tiny', pretrained=False, num_classes=4)
    elif "swin" in model_name:
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=4)
    elif "coatnet" in model_name:
        model = timm.create_model('coatnet_0_rw_224.sw_in1k', pretrained=False, num_classes=4)
    elif "efficientformer" in model_name:
        model = timm.create_model('efficientformer_l1', pretrained=False, num_classes=4)
    elif "levit" in model_name:
        model = timm.create_model('levit_128s', pretrained=False, num_classes=4)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

# Updated local paths (using raw strings for Windows paths)
checkpoint_7200_path = r'C:\Final_Year_Project\checkpoint-7200'
model_paths = [
    r'C:\Final_Year_Project\Copy_of_coatnet_epoch_30.pth',
    r'C:\Final_Year_Project\Copy_of_ConvNeXt_model_epoch_29.pt',
    r'C:\Final_Year_Project\Copy_of_efficientformer_epoch_30.pth',
    r'C:\Final_Year_Project\Copy_of_swin_epoch_30.pth',
]

# Load Hugging Face model with local files
try:
    model_hf = AutoModelForImageClassification.from_pretrained(
        checkpoint_7200_path,
        local_files_only=True
    )
    preprocessor = AutoImageProcessor.from_pretrained(
        checkpoint_7200_path,
        local_files_only=True
    )
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")
    model_hf = None
    preprocessor = None

# Load PyTorch models
def load_pytorch_model(model_path):
    try:
        model_name = os.path.basename(model_path)  # Get filename only
        model = load_model_architecture(model_name)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

models_pt = [m for m in (load_pytorch_model(path) for path in model_paths) if m is not None]

# Combine models only if Hugging Face model loaded successfully
all_models = []
model_names = []
if model_hf:
    all_models.append(model_hf)
    model_names.append("Swin (HF)")
    
model_names += ["CoAtNet", "ConvNeXt", "EfficientFormer", "Swin"]
all_models += models_pt

# Image preprocessing
def preprocess_image(image_file, target_size=(224, 224)):
    try:
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB').resize(target_size)
        
        # Hugging Face preprocessing
        inputs_hf = preprocessor(images=img, return_tensors="pt") if preprocessor else None
        
        # PyTorch preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_pt = transform(img).unsqueeze(0)
        
        return inputs_hf, img_pt
    except Exception as e:
        print(f"Image processing error: {e}")
        return None, None

# Prediction function
def get_predictions(model, inputs, is_huggingface=True):
    try:
        with torch.no_grad():
            if is_huggingface and inputs:
                outputs = model(**inputs)
                logits = outputs.logits
            else:
                logits = model(inputs)
            
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            return predicted_class.item(), confidence.item(), probabilities.squeeze().tolist()
    except Exception as e:
        print(f"Prediction error: {e}")
        return -1, 0.0, [0.0]*4

# Class labels
class_labels = ["cataract", "glaucoma", "diabetic_retinopathy", "normal"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        file = request.files['file']
        inputs_hf, img_pt = preprocess_image(file)
        
        if inputs_hf is None or img_pt is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
        results = []
        for i, (model, name) in enumerate(zip(all_models, model_names)):
            if i == 0 and model_hf:  # HF model
                pred_class, conf, probs = get_predictions(model, inputs_hf, True)
            else:
                pred_class, conf, probs = get_predictions(model, img_pt, False)
            
            results.append({
                'model': name,
                'prediction': class_labels[pred_class] if pred_class != -1 else 'error',
                'confidence': round(conf, 4),
                'probabilities': {k: round(v, 4) for k, v in zip(class_labels, probs)}
            })
            
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)