import torch

checkpoint = torch.load("best_efficientvit_performer.pth", map_location='cpu')

print("Available keys:")
for key in checkpoint:
    print(f"{key} → type: {type(checkpoint[key])}, length: {len(checkpoint[key]) if hasattr(checkpoint[key], '__len__') else 'N/A'}")

# Optionally print sample values
if 'train_loss' in checkpoint:
    print("Sample train_loss:", checkpoint['train_loss'][:5])
if 'val_accuracy' in checkpoint:
    print("Sample val_accuracy:", checkpoint['val_accuracy'][:5])
