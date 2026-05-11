"""
Galaxy Zoo 2 Classification - Inference Script

Load trained GalaxyResNet model and make predictions on galaxy images.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class GalaxyResNet(nn.Module):
    """ResNet18-based transfer learning model for galaxy morphology classification"""
    
    def __init__(self, num_classes=3):
        super(GalaxyResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze early layers (layer1, layer2)
        for name, param in self.backbone.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False
        
        # Replace final classifier
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def load_model(checkpoint_path='./models/best_model.pth', device='cuda'):
    """Load trained GalaxyResNet model from checkpoint"""
    model = GalaxyResNet(num_classes=3)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path):
    """Preprocess galaxy image for model input (validation transforms)"""
    transform = transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ImageNet normalization (used in ResNet18 pretraining)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def predict(model, image_tensor, device='cuda'):
    """Make prediction on preprocessed image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    return probs


def visualize_prediction(image_path, probs):
    """Visualize image with prediction probabilities"""
    class_names = ['Smooth', 'Featured/Disk', 'Artifact']
    
    # Load and display image
    image = Image.open(image_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Show image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Galaxy Image', fontsize=14, fontweight='bold')
    
    # Show probabilities
    colors = ['skyblue', 'coral', 'lightgreen']
    bars = ax2.barh(class_names, probs, color=colors, edgecolor='black')
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_title('Morphology Classification', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add probability values on bars
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1%}', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\nClassification Results:")
    print("=" * 40)
    for name, prob in zip(class_names, probs):
        print(f"{name:15s}: {prob:6.2%}")
    print("=" * 40)
    print(f"Predicted Class: {class_names[np.argmax(probs)]}")


def main():
    """Example usage of galaxy morphology classification inference"""
    # Configuration
    checkpoint_path = './models/best_model.pth'
    image_path = 'path/to/your/galaxy_image.jpg'  # Update with actual image path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading GalaxyResNet model from {checkpoint_path}")
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Preprocess and predict
    image_tensor = preprocess_image(image_path)
    probs = predict(model, image_tensor, device)
    
    # Visualize results
    visualize_prediction(image_path, probs)


if __name__ == '__main__':
    main()
