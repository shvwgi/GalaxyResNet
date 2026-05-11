"""
Galaxy Zoo 2 Classification - Inference Script

Load trained model and make predictions on new galaxy images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class GalaxyCNN(nn.Module):
    """Custom CNN for galaxy morphology classification"""
    
    def __init__(self, num_classes=3):
        super(GalaxyCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flatten_size = 256 * 14 * 14

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


def load_model(checkpoint_path='./models/best_model.pth', device='cuda'):
    """Load trained model from checkpoint"""
    model = GalaxyCNN(num_classes=3)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path):
    """Preprocess galaxy image for model input"""
    transform = transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
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
    """Example usage"""
    # Configuration
    checkpoint_path = './models/best_model.pth'
    image_path = 'path/to/your/galaxy_image.jpg'  # Update this
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model from {checkpoint_path}")
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
