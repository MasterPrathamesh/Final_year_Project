import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
MODEL_PATH = "models/Coatnet_epoch_30.pth"
ID2LABEL = {
    0: "cataract",
    1: "diabetic_retinopathy",
    2: "glaucoma",
    3: "normal"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

class CoatNetModel:
    def __init__(self):
        self.model = None
        self.transform = None
        self.load_model()

    def load_model(self):
        # Initialize the model architecture
        self.model = timm.create_model('coatnet_0_rw_224.sw_in1k', pretrained=False, num_classes=4)
        
        # Load the checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            predicted_class_name = ID2LABEL[predicted_class]
        
        return predicted_class_name, confidence

# Initialize model when module is imported
coatnet_model = CoatNetModel()