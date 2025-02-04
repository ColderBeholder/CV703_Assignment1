### Importing Required Files and Libraries ###

import torch
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102
import torch.nn as nn
import torch.optim as optim
import time
from cnnHybrid import ConvNeXtWithTransformer
from utils import dataloader, START_seed, seed_worker, train_one_epoch, validate


### Initialization and Seeding ###

torch.cuda.empty_cache()

g = torch.Generator()
g.manual_seed(10)
START_seed()


### Preparing the Data ###

flowers102_root = './data/flowers-102'

train_batch_size = 64
test_batch_size = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

flowers_train = Flowers102(root=flowers102_root, split='test', download=True, transform=train_transform)
flowers_test = Flowers102(root=flowers102_root, split='train', download=True, transform=val_transform)

train_dataloader = dataloader(flowers_train, train_batch_size, 
                              device, seed_worker, g)

val_dataloader = dataloader(flowers_test, test_batch_size, 
                              device, seed_worker, g)


### Creating the Model and Setting Architecture Parameters ###

num_classes = 102
embed_dim = 512
nhead = 16
num_transformer_layers = 1

model = ConvNeXtWithTransformer(num_classes=num_classes, embed_dim=embed_dim, nhead=nhead, num_transformer_layers=num_transformer_layers).to(device)


### Setting Hyperparameters ###

lr = 1e-4
epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


### Training and Validation ###

start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    train_loss, train_accuracy = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)
    scheduler.step()

print("Training complete!")

b_time = time.time() - start_time
print(f"Training Time: {b_time} seconds")