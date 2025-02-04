import torch
import numpy as np
import random
from tqdm import tqdm


def dataloader(data, batch_size, device, seed_worker, g):
    return torch.utils.data.DataLoader(
    data, batch_size=batch_size, shuffle=True,
    pin_memory=True, pin_memory_device=device, num_workers=2,
    persistent_workers=True, prefetch_factor=2,
    worker_init_fn=seed_worker,generator=g
)

def START_seed():
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = 10
    np.random.seed(worker_seed+worker_id)
    random.seed(worker_seed+worker_id)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, ascii=True, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, ascii=True, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    print(f"Val Loss: {epoch_loss:.4f}, Val Accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy