import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from efficientvit_with_performer import EfficientViT

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    train_dir = os.path.join("..", 'dataset', 'imagenet100', 'train')
    val_dir = os.path.join("..", 'dataset', 'imagenet100', 'val')

    # Transforms
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

    # DataLoaders
    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    valset = datasets.ImageFolder(val_dir, transform=transform_val)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

    # Model setup
    model = EfficientViT(img_size=224, num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    best_acc = 0.0

    # For plotting
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(50):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/50]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        scheduler.step()

        # ✅ Save training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100. * val_correct / val_total
        val_accuracies.append(val_acc)

        print(f"✅ Epoch [{epoch+1}/50]: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.join(root_dir, "outputs"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(root_dir, "outputs", "best_efficientvit_performer.pth"))
            print(f"💾 Model saved with best accuracy {best_acc:.2f}%")

    print("🎯 Training complete.")

    # ✅ Plotting the learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(root_dir, 'outputs', 'learning_curve.png'))
    print("📈 Saved learning curve to 'outputs/learning_curve.png'")

if __name__ == '__main__':
    main()
