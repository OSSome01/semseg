import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data_prep import RUGD
from unet import UNet

# Define the device to be used for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the transforms to be applied to the data
transform = transforms.Compose([
    transforms.Resize((256, 256)),    # Resize the image to 256x256
    transforms.ToTensor(),            # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # Normalize the pixel values to [-1, 1]
])

# Load the RUGD dataset
trainset = RUGD(root_dir='/home/ossome/semantic_segmentation/sem_seg/RUGD', split='train', transform=transform)
testset = RUGD(root_dir='/home/ossome/semantic_segmentation/sem_seg/RUGD', split='test', transform=transform)

# Create data loaders for the datasets
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# Define the model
model = UNet(num_channels=3, num_classes=25)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Train the model on the training set
    model.train()
    train_loss = 0.0
    for images, labels in trainloader:
        print('images', images.shape)
        print('labels', labels.shape)
        images = images.to(device)
        labels = nn.functional.one_hot(labels.long(), 25)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        
    train_loss /= len(trainloader.dataset)
    
    # Test the model on the test set
    model.eval()
    test_loss = 0.0
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        test_loss += loss.item() * images.size(0)
        
    test_loss /= len(testloader.dataset)
    
    # Print the loss for the current epoch
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')