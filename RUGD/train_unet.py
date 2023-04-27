import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from data_prep import RUGD
from models import UNet, AttnUNet, ViT_Segmentation
from segformer import SegFormer
# import wandb


# wandb.init(project='sem_seg', 
#             entity='ossome',
#             config={'learning_rate':0.01,
#                     'scheduler_stepsize':5,
#                     'schduler_gamma':0.1,
#                     'epochs':15})

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),    
        transforms.ToTensor(),           
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
    ])

   # Load the RUGD dataset
    trainset = RUGD(root_dir='/home/ossome/semantic_segmentation/sem_seg/RUGD', split='train', transform=transform)
    valset = RUGD(root_dir='/home/ossome/semantic_segmentation/sem_seg/RUGD', split='val', transform=transform)
    testset = RUGD(root_dir='/home/ossome/semantic_segmentation/sem_seg/RUGD', split='test', transform=transform)
    
    # Create data loaders for the datasets
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=2, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=2)

    model = SegFormer(input_channels=3, num_classes=21)
    # model = ViT_Segmentation(image_size=256, patch_size=16, num_classes=25, dim=1024, depth=6, heads=16, mlp_dim=2048)
    # model = AttnUNet(3, 25)
    model.to(device)
    # model.load_state_dict(torch.load('best_model_weights.pth'))

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.90)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    num_epochs = 15
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(trainloader):
            #labels = labels.permute(0,3,1,2)
            # print('labels_1', labels.shape)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).to(device)
            # print('output',outputs.requires_grad_())
            outputs = nn.functional.softmax(outputs.float(), dim=1)  
            _, predicted = torch.max(outputs, dim=1)  
            # print('predicted',type(predicted))
            # print('predicted', predicted.shape)
            # print('labels', labels.shape)
            # print('labels', type(labels))
            loss = criterion(outputs, (labels.type(torch.LongTensor)).to(device))
            # loss = criterion(predicted.type(torch.float32), labels.type(torch.float32))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(trainloader.dataset)
        
        model.eval()
        
        val_loss = 0.0
        for images, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(images)
                outputs = nn.functional.softmax(outputs, dim=1)  
                _, predicted = torch.max(outputs, dim=1)  
                loss = criterion(outputs, (labels.type(torch.LongTensor)).to(device)) 
            
            val_loss += loss.item() * images.size(0)
            
        val_loss /= len(valloader.dataset)
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), 'best_model_weights.pth')
        # wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        scheduler.step()

        # Print the loss for the current epoch
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')