import torch
import torch.nn as nn

def accuracy(data_loader, model):
    correct = 0
    total = 0
    running_loss = 0
    n = len(data_loader)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        loss_result = running_loss / n

    acc = 100 * correct / total
    model.train()
    return acc, loss_result