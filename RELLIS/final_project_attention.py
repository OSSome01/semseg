# -*- coding: utf-8 -*-
import os
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import torch
import torch.nn as nn

class RUGD(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform


        # Define the classes and color map
        self.classes = ['void', 'dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object', 'asphalt', 'building',
                        'log', 'person', 'fence', 'bush', 'concrete', 'barrier','uphill','downhill','puddle','mud','rubble']

        self.color_map = {
            (0, 0, 0): 0,   
            (108, 64, 20): 1,       
            (0, 102, 0): 2,       
            (0, 255, 0): 3,
            (0, 153, 153): 4,
            (0, 128, 255): 5,
            (0, 0, 255): 6,
            (255, 255, 0): 7,
            (255, 0, 127): 8,
            (64, 64, 64): 9,
            (255, 0, 0): 10,
            (102, 0, 0): 11,
            (204, 153, 255): 12,
            (102, 0, 204): 13,
            (255, 153, 204): 14,
            (170, 170, 170): 15,
            (41, 121, 255): 16,
            (101, 31, 255): 17,
            (137, 149, 9): 18,
            (134, 255, 239): 19,
            (99, 66, 34): 20,
            (110, 22, 138): 21,
        }
        # Load the list of image and label filenames

        self.image_filenames = sorted(glob.glob(os.path.join(self.root_dir + self.split+'/images/', '*.jpg')))
        self.label_filenames = sorted(glob.glob(os.path.join(self.root_dir + self.split+'/labels/', '*.png')))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.image_filenames[idx]).convert('RGB')
        # Apply the transforms to the image
        if self.transform is not None:
            image = self.transform(image)

        # Load the label
        label = Image.open(self.label_filenames[idx]).convert('RGB')
        label = label.resize((128,128), resample=Image.NEAREST)
        label_classes = np.zeros((128*128), dtype=np.uint8)
        label = np.array(label)
        label_pixels = label.reshape(-1,3)
        for i in range(label_pixels.shape[0]):
          rgb = tuple(label_pixels[i])
          if rgb in self.color_map:
            label_classes[i] = self.color_map[rgb]
        label = label_classes.reshape(128,128)        

        return image, label

class LongTailLoss(nn.Module):
    def __init__(self, num_classes, beta=0.5):
        super(LongTailLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta

    def forward(self, inputs, targets):
        # Flatten inputs and targets to have the same batch size
        inputs = inputs.view(-1, self.num_classes)
        targets = targets.view(-1)

        class_counts = torch.bincount(targets, minlength=self.num_classes).to(device)
        present_classes = torch.nonzero(class_counts).squeeze().to(device)
        absent_classes = torch.nonzero(class_counts == 0).squeeze().to(device)
        
        # Compute class weights only for present classes
        present_class_counts = class_counts[present_classes].float()
        present_class_weights = (1 - self.beta) / (1 - self.beta ** present_class_counts)
        present_class_weights /= torch.sum(present_class_weights) / len(present_classes)
        
        # Concatenate present class weights with zero weights for absent classes
        class_weights = torch.zeros(self.num_classes).to(device)
        class_weights[present_classes] = present_class_weights
        
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(inputs, targets)

        return loss

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        x = self.mpconv(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.attention = AttentionBlock(in_channels // 2, in_channels // 2, in_channels // 4)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.attention(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(num_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, num_classes)
    
    def forward(self, x):
        x1 = self.inc(x).to('cuda')
        x2 = self.down1(x1).to('cuda')
        x3 = self.down2(x2).to('cuda')
        x4 = self.down3(x3).to('cuda')
        x5 = self.down4(x4).to('cuda')
        x = self.up1(x5, x4).to('cuda')
        x = self.up2(x, x3).to('cuda')
        x = self.up3(x, x2).to('cuda')
        x = self.up4(x, x1).to('cuda')
        x = self.outc(x).to('cuda')
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    transform = transforms.Compose([
        transforms.Resize((128,128)),    # Resize the image to 256x256
        transforms.ToTensor(),            # Convert the image to a PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # Normalize the pixel values to [-1, 1]
    ])


    trainset = RUGD(root_dir='/home/bhanushali.r/Rellis-3D/Rellis-3D/', split='train', transform=transform)
    valset = RUGD(root_dir='/home/bhanushali.r/Rellis-3D/Rellis-3D/', split='val', transform=transform)
    testset = RUGD(root_dir='/home/bhanushali.r/Rellis-3D/Rellis-3D/', split='test', transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=16)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=16)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=16)

    model = UNet(num_channels=3, num_classes=22)
    try:
        model.load_state_dict(torch.load("/home/bhanushali.r/Rellis-3D/Rellis-3D/rellis_lowest_.pth", map_location=device))
    except:
        pass
    model.to(device)

    #criterion = nn.CrossEntropyLoss()
    criterion = criterion = LongTailLoss(num_classes=22)    
    optimizer = optim.Adam(model.parameters(), lr=0.006)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
    # Train the model
    num_epochs = 10
    best_val_loss = float('inf')
    wandb.init(project="Rellis-3D")

    # Define the hyperparameters to be logged
    config = wandb.config
    config.learning_rate = 0.008
    config.momentum = 0.95
    config.batch_size = 16
    config.num_epochs = 10
    
    wandb.watch(model)


    for epoch in range(num_epochs):
        # Train the model on the training set
        model.train()
        train_loss = 0.0
        i=1
        for images, labels in tqdm(trainloader):
            #labels = labels.permute(0,3,1,2)
            # print('labels_1', labels.shape)
            images = images.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            
            outputs = model(images).to(device)
            # print('output',outputs.requires_grad_())
            outputs = nn.functional.softmax(outputs.float(), dim=1)  # apply softmax to output
            _, predicted = torch.max(outputs, dim=1)  # get predicted class along channel dimension
            loss = criterion(outputs, (labels.type(torch.LongTensor)).to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            wandb.log({'step_loss': train_loss/(i*images.size(0)), 'learning_rate': scheduler.get_last_lr()})
            i+=1
            
        train_loss /= len(trainloader.dataset)
        
        # Test the model on the val set
        model.eval()
        
        val_loss = 0.0
        for images, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(images)
                outputs = nn.functional.softmax(outputs, dim=1)  # apply softmax to output
                _, predicted = torch.max(outputs, dim=1)  # get predicted class along channel dimension
                loss = criterion(outputs, (labels.type(torch.LongTensor)).to(device))  # use predicted class as label
            
            val_loss += loss.item() * images.size(0)
            

            
        val_loss /= len(valloader.dataset)
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch+1})
        torch.save(model.state_dict(), 'rellis_'+str(val_loss)+'_.pth')
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), 'rellis_lowest_128.pth')
          try:
              wandb.run.summary['best_val_loss'] = val_loss
          except:
              pass
        try:
            wandb.save('rellis_final_attn_wb.pth')
            wandb.save('best_model_weights_final.pth')
        except:
            pass
        
        scheduler.step()

        # Print the loss for the current epoch
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    try:
        wandb.finish()
    except:
        pass
    torch.save(model.state_dict(), '/home/bhanushali.r/Rellis-3D/Rellis-3D/rellis_final_attn.pth')