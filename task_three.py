### Importing Required Files and Libraries ###

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, FGVCAircraft, Flowers102
from torch.utils.data import ConcatDataset
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

fgvc_aircraft_root = './data'
imagewoof_root = './data/imagewoof2-160'
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

fgvc_trainval = FGVCAircraft(root=fgvc_aircraft_root, split='trainval', download=True, transform=train_transform)
fgvc_test = FGVCAircraft(root=fgvc_aircraft_root, split='test', download=True, transform=val_transform)

flowers_train = Flowers102(root=flowers102_root, split='test', download=True, transform=train_transform)
flowers_test = Flowers102(root=flowers102_root, split='train', download=True, transform=val_transform)

train_dir = os.path.join(imagewoof_root, 'imagewoof2-160/train')
valid_dir = os.path.join(imagewoof_root, 'imagewoof2-160/val')

imagewoof_train = ImageFolder(root=train_dir, transform=train_transform)
imagewoof_val = ImageFolder(root=valid_dir, transform=val_transform)

def update_targets(dataset, start_label):
    dataset.targets = [label for _, label in dataset]
    unique_labels = set(dataset.targets)

    return dataset, start_label + len(unique_labels)

start_label = 0

imagewoof_train, start_label_fgcv = update_targets(imagewoof_train, start_label)
imagewoof_val, _ = update_targets(imagewoof_val, start_label)

class ModifiedFGVCAircraft(FGVCAircraft):
    def __init__(self, root, split='trainval', download=False, transform=None, startlabel=0):
        super(ModifiedFGVCAircraft, self).__init__(root=root, split=split, download=download, transform=transform)
        self.startlabel = startlabel
    def __getitem__(self, index):
        image, label = super(ModifiedFGVCAircraft, self).__getitem__(index)
        label += self.startlabel
        return image, label
fgvc_trainval = ModifiedFGVCAircraft(root=fgvc_aircraft_root, split='trainval', download=True, transform=train_transform, startlabel=start_label_fgcv)
fgvc_test = ModifiedFGVCAircraft(root=fgvc_aircraft_root, split='test', download=True, transform=val_transform, startlabel=start_label_fgcv)

fgvc_trainval, start_label_flowers = update_targets(fgvc_trainval, start_label_fgcv)
fgvc_test, _ = update_targets(fgvc_test, start_label_fgcv)


class ModifiedFlowers102(Flowers102):
    def __init__(self, root, split='train', download=False, transform=None, startlabel=0):
        super(ModifiedFlowers102, self).__init__(root=root, split=split, download=download, transform=transform)
        self.startlabel = startlabel
    def __getitem__(self, index):
        image, label = super(ModifiedFlowers102, self).__getitem__(index)
        label += self.startlabel
        return image, label
flowers_train = ModifiedFlowers102(root=flowers102_root, split='test', download=True, transform=train_transform, startlabel=start_label_flowers)
flowers_test = ModifiedFlowers102(root=flowers102_root, split='train', download=True, transform=val_transform, startlabel=start_label_flowers)

flowers_train, _ = update_targets(flowers_train, start_label_flowers)
flowers_test, _ = update_targets(flowers_test, start_label_flowers)

train_dataset = ConcatDataset([imagewoof_train, fgvc_trainval, flowers_train])
test_dataset = ConcatDataset([imagewoof_val, fgvc_test, flowers_test])

train_dataloader = dataloader(train_dataset, train_batch_size, 
                              device, seed_worker, g)

val_dataloader = dataloader(test_dataset, test_batch_size, 
                              device, seed_worker, g)


### Creating the Model and Setting Architecture Parameters ###

num_classes = 212
embed_dim = 512
nhead = 16
num_transformer_layers = 1

model = ConvNeXtWithTransformer(num_classes=num_classes).to(device)


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