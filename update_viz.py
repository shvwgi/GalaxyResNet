import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from galaxy_datasets.pytorch import GZ2
from sklearn.metrics import mean_absolute_error
import os

# Model definition
class GalaxyCNN(nn.Module):
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

# Load dataset
print("Loading dataset...")
data_transforms = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = GZ2(root='./data', train=True, download=False, transform=data_transforms)
pilot_size = 2000
pilot_dataset, _ = random_split(dataset, [pilot_size, len(dataset) - pilot_size])
train_size = int(0.8 * len(pilot_dataset))
val_size = len(pilot_dataset) - train_size
train_subset, val_subset = random_split(pilot_dataset, [train_size, val_size])
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# Load model
print("Loading trained model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GalaxyCNN(num_classes=3).to(device)
checkpoint = torch.load('./models/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate
print("Evaluating model performance...")
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch in val_loader:
        inputs = batch['image'].to(device)
        targets = torch.stack([
            batch['smooth-or-featured-gz2_smooth'],
            batch['smooth-or-featured-gz2_featured-or-disk'],
            batch['smooth-or-featured-gz2_artifact']
        ], dim=1).float()
        targets = targets / (targets.sum(dim=1, keepdim=True) + 1e-6)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).cpu()
        all_predictions.append(probs)
        all_targets.append(targets)

all_predictions = torch.cat(all_predictions, dim=0).numpy()
all_targets = torch.cat(all_targets, dim=0).numpy()

# Calculate MAE
class_names = ['Smooth', 'Featured/Disk', 'Artifact']
mae_per_class = [mean_absolute_error(all_targets[:, i], all_predictions[:, i]) for i in range(3)]

# Create improved visualization
print("Creating improved visualization...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, (class_name, mae) in enumerate(zip(class_names, mae_per_class)):
    if i == 2:  # Artifact - use hexbin with zoom
        hb = axes[i].hexbin(all_targets[:, i], all_predictions[:, i], 
                           gridsize=25, cmap='YlOrRd', mincnt=1, edgecolors='face', linewidths=0.2)
        cb = plt.colorbar(hb, ax=axes[i], label='Density', pad=0.02)
        axes[i].set_xlim(-0.005, 0.25)
        axes[i].set_ylim(-0.005, 0.25)
        max_val = 0.25
    else:  # Smooth and Featured
        hb = axes[i].hexbin(all_targets[:, i], all_predictions[:, i], 
                           gridsize=30, cmap='Blues', mincnt=1, edgecolors='face', linewidths=0.2)
        cb = plt.colorbar(hb, ax=axes[i], label='Density', pad=0.02)
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        max_val = 1.0
    
    # Perfect prediction line
    axes[i].plot([0, max_val], [0, max_val], 'r--', linewidth=2.5, 
                 label='Perfect Prediction', zorder=10, alpha=0.8)
    
    axes[i].set_xlabel(f'True {class_name} Probability', fontsize=11, fontweight='bold')
    axes[i].set_ylabel(f'Predicted {class_name} Probability', fontsize=11, fontweight='bold')
    axes[i].set_title(f'{class_name}\nMAE: {mae:.4f}', fontsize=13, fontweight='bold', pad=10)
    axes[i].grid(alpha=0.3, linestyle=':', linewidth=0.5)
    axes[i].legend(loc='upper left', fontsize=9, framealpha=0.9)
    axes[i].set_aspect('equal', adjustable='box')

plt.suptitle('Model Performance: Predicted vs True Probabilities', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('./plots/prediction_scatter.png', dpi=150, bbox_inches='tight')
print("✓ Saved to ./plots/prediction_scatter.png")
plt.show()

# Print statistics
print("\n" + "="*60)
print("PER-CLASS PERFORMANCE METRICS")
print("="*60)
for name, mae in zip(class_names, mae_per_class):
    print(f"{name:20s}: MAE = {mae:.4f}")

# Artifact analysis
artifact_true = all_targets[:, 2]
artifact_pred = all_predictions[:, 2]
high_artifact = (artifact_true > 0.1).sum()
print(f"\n{'Artifact Analysis':20s}")
print(f"{'  High artifact (>10%)':30s}: {high_artifact:>4d} samples")
print(f"{'  Max probability (true)':30s}: {artifact_true.max():>6.4f}")
print(f"{'  Max probability (pred)':30s}: {artifact_pred.max():>6.4f}")
print(f"{'  Mean probability (true)':30s}: {artifact_true.mean():>6.4f}")
print(f"{'  Mean probability (pred)':30s}: {artifact_pred.mean():>6.4f}")
print("="*60)
print("\n✓ Visualization complete! The artifact plot is now zoomed in.")
