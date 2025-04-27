import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from efficientvit_with_performer import EfficientViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset Paths and Transforms
# ----------------------------
train_dir = os.path.join('..', 'dataset', 'imagenet100', 'train')
val_dir = os.path.join('..', 'dataset', 'imagenet100', 'val')

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# ----------------------------
# Model, Loss, Optimizer
# ----------------------------
model = EfficientViT(img_size=224, num_classes=100)
model.to(device)

mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=100)
criterion = LabelSmoothingCrossEntropy()

optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
lr_scheduler = CosineLRScheduler(optimizer, t_initial=100)

writer = SummaryWriter(log_dir='./runs/efficientvit_performer')
lr_history = []
best_acc = 0

# ----------------------------
# Training Loop
# ----------------------------
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(targets, 1)
        correct += (predicted == labels).sum().item()
        total += targets.size(0)

    lr_scheduler.step(epoch)
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)

    writer.add_scalar("LR", current_lr, epoch)
    writer.add_scalar("Train/Loss", total_loss / len(train_loader), epoch)
    writer.add_scalar("Train/Accuracy", 100 * correct / total, epoch)

    # ----------------------------
    # Validation
    # ----------------------------
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == targets).sum().item()
            val_total += targets.size(0)

    val_acc = 100 * val_correct / val_total
    writer.add_scalar("Val/Accuracy", val_acc, epoch)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "efficientvit_performer_best_v2.pth")

# ----------------------------
# Save Final Model
# ----------------------------
torch.save(model.state_dict(), "efficientvit_performer_final_v2.pth")

# ----------------------------
# Plot LR Curve
# ----------------------------
plt.plot(lr_history)
plt.title("Learning Rate over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.savefig("lr_curve.png")
plt.show()
