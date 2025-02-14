
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from transformers import AutoModelForImageClassification, AutoImageProcessor
from safetensors.torch import load_file
from PIL import Image
import torch.nn.functional as F

  # Load model architecture based on the model name
  def load_model_architecture(model_name):
      if "ConvNeXt" in model_name:
          # Use timm's ConvNeXt
          model = timm.create_model('convnext_tiny', pretrained=False, num_classes=4)
      elif "swin" in model_name:
          # Use timm's Swin Transformer
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

  # Paths to checkpoints
  checkpoint_7200_path = 'C:\Users\rajkh\Desktop\Eye_D\All_Checkpoints\checkpoint-7200'
  model_paths = [
      'C:\Users\rajkh\Desktop\Eye_D\Copy_of_coatnet_epoch_30.pth',
      'C:\Users\rajkh\Desktop\Eye_D\All_Checkpoints\Copy_of_ConvNeXt_model_epoch_29.pt',
      'C:\Users\rajkh\Desktop\Eye_D\All_Checkpoints\Copy_of_efficientformer_epoch_30.pth',
      'C:\Users\rajkh\Desktop\Eye_D\All_Checkpoints\Copy_of_swin_epoch_30.pth',
  ]

  # Load Hugging Face model (assuming it's a Swin model)
  model_hf = AutoModelForImageClassification.from_pretrained(checkpoint_7200_path)
  preprocessor = AutoImageProcessor.from_pretrained(checkpoint_7200_path)

  # Load PyTorch models (state_dict only)
  def load_pytorch_model(model_path):
      model_name = model_path.split('/')[-1]
      model = load_model_architecture(model_name)
      checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
      if "model_state_dict" in checkpoint:
          model.load_state_dict(checkpoint["model_state_dict"], strict=False)
      else:
          model.load_state_dict(checkpoint, strict=False)
      model.eval()
      return model

  models_pt = [load_pytorch_model(path) for path in model_paths]

  # Combine all models and correct model names
  all_models = [model_hf] + models_pt
  model_names = ["Levit", "CoAtNet", "ConvNeXt", "EfficientFormer", "Swin"]  # Adjusted names based on actual models

  # Image preprocessing remains the same
  def preprocess_image(image_path, target_size=(224, 224)):
      img = Image.open(image_path).resize(target_size)
      inputs_hf = preprocessor(img, return_tensors="pt")
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      img_pt = transform(img).unsqueeze(0)
      return inputs_hf, img_pt

  # Prediction function remains the same
  def get_predictions(model, inputs, is_huggingface=True):
      with torch.no_grad():
          if is_huggingface:
              outputs = model(**inputs)
              logits = outputs.logits
          else:
              logits = model(inputs)

          probabilities = F.softmax(logits, dim=1)
          confidence, predicted_class = torch.max(probabilities, 1)

      return predicted_class.item(), confidence.item(), probabilities.squeeze().tolist()

  # Define class labels
  class_labels = ["cataract", "glaucoma", "diabetic_retinopathy", " "]

  # Image path for testing
  test_image_path = 'C:\Users\rajkh\Desktop\Eye_D\cataract_2.jpg'
  inputs_hf, img_pt = preprocess_image(test_image_path)

  # Get predictions from all models
  for i, (model, name) in enumerate(zip(all_models, model_names)):
      if i == 0:
          # Hugging Face model
          predicted_class, confidence, probabilities = get_predictions(model, inputs_hf, is_huggingface=True)
      else:
          # PyTorch models
          predicted_class, confidence, probabilities = get_predictions(model, img_pt, is_huggingface=False)

      print(f"\nModel: {name}")
      print(f"Predicted Class: {class_labels[predicted_class]} (Confidence: {confidence:.4f})")
      print("Class Probabilities:")
      for label, prob in zip(class_labels, probabilities):
          print(f"  {label}: {prob:.4f}")
