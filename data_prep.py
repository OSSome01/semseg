import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob

class RUGD(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform


        # Define the classes and color map
        self.classes = ['void', 'dirt', 'sand', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'container/generic-object', 'asphalt', 'gravel', 'building', 'mulch', 'rock-bed',
                        'log', 'bicycle', 'person', 'fence', 'bush', 'sign', 'rock-bed', 'bridge', 'concrete', 'picnic-table']
        self.color_map = {
            (0, 0, 0): 0,   
            (108, 64, 20): 1,    
            (255, 229, 204): 2,   
            (0, 102, 0): 3,       
            (0, 255, 0): 4,
            (0, 153, 153): 5,
            (0, 128, 255): 6,
            (0, 0, 255): 7,
            (255, 255, 0): 8,
            (255, 0, 127): 9,
            (64, 64, 64): 10,
            (255, 128, 0): 11,
            (255, 0, 0): 12,
            (153, 76, 0): 13,
            (102, 102, 0): 14,
            (102, 0, 0): 15,
            (0, 255, 128): 16,
            (204, 153, 255): 17,
            (102, 0, 204): 18,
            (255, 153, 204): 19,
            (0, 102, 102): 20,
            (153, 204, 255): 21,
            (102, 255, 255): 22,
            (101, 101, 11): 23,
            (114, 85, 47): 24,
        }

        # Load the list of image and label filenames

        self.image_filenames = sorted(glob.glob(os.path.join(self.root_dir, self.split, 'images', '*.png')))
        self.label_filenames = sorted(glob.glob(os.path.join(self.root_dir, self.split, 'masks', '*.png')))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load the image and label


        image = Image.open(self.image_filenames[idx]).convert('RGB')
        label = Image.open(self.label_filenames[idx]).convert('RGB')

        # Convert the label to a numpy array and apply the color map
        label = np.array(label)
        label = self.color_map[tuple(label.reshape(-1, 3)[0])]
        print('label', torch.tensor(label).shape)

        # Apply the transforms to the image and label
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        print('label', torch.tensor(label).shape)
        return image, label

# # Define the transforms to be applied to the data
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),    # Resize the image to 256x256
#     transforms.ToTensor(),            # Convert the image to a PyTorch tensor
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # Normalize the pixel values to [-1, 1]
# ])

# # Load the RUGD dataset
# trainset = RUGD(root_dir='/home/ossome/semantic_segmentation/sem_seg/RUGD', split='train', transform=transform)
# testset = RUGD(root_dir='/home/ossome/semantic_segmentation/sem_seg/RUGD', split='test', transform=transform)



# # Create data loaders for the datasets
# trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
# testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# # Print the number of images in the datasets
# print('Number of training images:', len(trainset))
# print('Number of test images:', len(testset))