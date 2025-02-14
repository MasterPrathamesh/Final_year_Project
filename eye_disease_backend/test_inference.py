from convnext_model import convnext_model
from swin_model import swin_model
from coatnet_model import coatnet_model
from efficientformer_model import efficientformer_model

def run_inference(image_path):
    """
    Run inference using multiple models
    """
    results = {}
    
    # ConvNeXt prediction
    print("\nRunning ConvNeXt model inference...")
    conv_class, conv_conf = convnext_model.predict(image_path)
    results["ConvNeXt"] = {"class": conv_class, "confidence": conv_conf}
    
    # Swin prediction
    print("\nRunning Swin model inference...")
    swin_class, swin_conf = swin_model.predict(image_path)
    results["Swin"] = {"class": swin_class, "confidence": swin_conf}
    
    # CoatNet prediction
    print("\nRunning CoatNet model inference...")
    coat_class, coat_conf = coatnet_model.predict(image_path)
    results["CoatNet"] = {"class": coat_class, "confidence": coat_conf}
    
    # EfficientFormer prediction
    print("\nRunning EfficientFormer model inference...")
    eff_class, eff_conf = efficientformer_model.predict(image_path)
    results["EfficientFormer"] = {"class": eff_class, "confidence": eff_conf}
    
    return results

def print_model_results(results):
    """
    Print formatted results for all models
    """
    print("\nResults:")
    print("-" * 50)
    for model_name, result in results.items():
        print(f"{model_name} Model:")
        print(f"Predicted Class: {result['class']}")
        print(f"Confidence Score: {result['confidence']:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    try:
        # Path to test image
        image_path = "2466_right.jpg"
        
        # Perform predictions with all models
        print("Performing inference with multiple models...")
        results = run_inference(image_path)
        
        # Print results
        print_model_results(results)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise  # Re-raise the exception for debugging purposes