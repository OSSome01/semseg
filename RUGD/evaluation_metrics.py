import torch
import numpy as np
from tqdm import tqdm
import sys
import multiprocessing
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
from models import UNet, AttnUNet
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

        self.image_filenames = sorted(glob.glob(os.path.join(self.root_dir + self.split, '*.png')))
        self.label_filenames = sorted(glob.glob(os.path.join(self.root_dir + self.split+'_labels/', '*.png')))

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
        label = label.resize((256,256), resample=Image.NEAREST)
        label_classes = np.zeros((256*256), dtype=np.uint8)
        label = np.array(label)
        label_pixels = label.reshape(-1,3)
        for i in range(label_pixels.shape[0]):
          rgb = tuple(label_pixels[i])
          if rgb in self.color_map:
            label_classes[i] = self.color_map[rgb]
        label = label_classes.reshape(256,256)

        return image, label


def calculate_iou(outputs, labels, num_classes):
    iou_list = []
    for cls in range(num_classes):
        intersection = np.sum((outputs == cls) * (labels == cls))
        union = np.sum((outputs == cls) + (labels == cls))
        if union == 0:
            iou_list.append(0)
        else:
            iou_list.append(intersection / union)
    return iou_list

def calculate_metrics(outputs, labels, num_classes):
    outputs = outputs.flatten()
    labels = labels.flatten()

    # Calculate confusion matrix
    cm = confusion_matrix(labels, outputs, labels=list(range(num_classes)))

    # Calculate IoU
    iou_list = calculate_iou(outputs, labels, num_classes)
    miou = np.mean(iou_list)

    # Calculate pixel accuracy
    pixel_acc = np.diag(cm).sum() / cm.sum()

    # Calculate F1 score and recall
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    # f1_list = np.nan_to_num((2 * tp) / (2 * tp + fp + fn), nan=0, posinf=0, neginf=0)
    # recall_list = np.nan_to_num(tp / (tp + fn), nan=0, posinf=0, neginf=0)
    f1_list = np.nan_to_num((2 * tp) / (2 * tp + fp + fn), nan=0, posinf=np.finfo(np.float32).max, neginf=0)
    recall_list = np.nan_to_num(tp / (tp + fn), nan=0, posinf=np.finfo(np.float32).max, neginf=0)
    f1 = np.mean(f1_list)
    recall = np.mean(recall_list)

    return miou, pixel_acc, f1, recall
    # return miou, pixel_acc

if __name__ == '__main__':


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = RUGD(root_dir='/home/ossome/semantic_segmentation/sem_seg/RUGD/', split='test', transform=transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttnUNet(3, 25)
    model.load_state_dict(torch.load('/home/ossome/semantic_segmentation/sem_seg/attn_checkpoints/best_after_13epochs_train24618_val24642.pth', map_location=device))

    model.to(device)
    model.eval()

    test_dataloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)
    # Evaluate the model on the test dataset
    miou_list = []
    pixel_acc_list = []
    f1_list = []
    recall_list = []

    with torch.no_grad():
        for images,labels in tqdm(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)
            # _, predicted = torch.max(outputs, dim=1)
            # predicted_image = predicted[0].cpu().numpy()

            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()

            miou, pixel_acc, f1, recall = calculate_metrics(outputs, labels, 25)
            # miou, pixel_acc = calculate_metrics(outputs, labels, 22)

            miou_list.append(miou)
            pixel_acc_list.append(pixel_acc)
            f1_list.append(f1)
            recall_list.append(recall)

    avg_miou = np.mean(miou_list)
    avg_pixel_acc = np.mean(pixel_acc_list)
    avg_f1 = np.mean(f1_list)
    avg_recall = np.mean(recall_list)

    print(f'Average MIOU: {avg_miou:.4f}')
    print(f'Average pixel accuracy: {avg_pixel_acc:.4f}')
    print(f'Average F1 score: {avg_f1:.4f}')
    print(f'Average recall: {avg_recall:.4f}')