import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import warnings
from transformers import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Constants
MODEL_PATH = "models/ConvNeXt_model_epoch_29.pt"
ID2LABEL = {
    0: "cataract",
    1: "diabetic_retinopathy",
    2: "glaucoma",
    3: "normal"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

class ConvNextModel:
    def __init__(self):
        self.model = None
        self.image_processor = None
        self.load_model()

    def load_model(self):
        # Initialize the image processor
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        
        # Initialize the model architecture
        self.model = AutoModelForImageClassification.from_pretrained(
            "facebook/convnext-tiny-224",
            num_labels=4,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True
        )
        
        # Load the checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, image_path):
        """
        Predict the eye disease class for a given image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            tuple: (predicted_class_name, confidence_score)
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(image, return_tensors="pt")
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            predicted_class_name = self.model.config.id2label[predicted_class]
        
        return predicted_class_name, confidence

# Initialize model when module is imported
convnext_model = ConvNextModel()