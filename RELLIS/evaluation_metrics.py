import torch
import numpy as np
from tqdm import tqdm
from final_project_attention import UNet
import sys
import multiprocessing
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class RUGD(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform


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


        self.image_filenames = sorted(glob.glob(os.path.join(self.root_dir + self.split+'/images/', '*.jpg')))
        self.label_filenames = sorted(glob.glob(os.path.join(self.root_dir + self.split+'/labels/', '*.png')))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        image = Image.open(self.image_filenames[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)


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


    cm = confusion_matrix(labels, outputs, labels=list(range(num_classes)))


    iou_list = calculate_iou(outputs, labels, num_classes)
    miou = np.mean(iou_list)

    pixel_acc = np.diag(cm).sum() / cm.sum()


    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    f1_list = np.nan_to_num((2 * tp) / (2 * tp + fp + fn), nan=0, posinf=np.finfo(np.float32).max, neginf=0)
    recall_list = np.nan_to_num(tp / (tp + fn), nan=0, posinf=np.finfo(np.float32).max, neginf=0)
    f1 = np.mean(f1_list)
    recall = np.mean(recall_list)

    return miou, pixel_acc, f1, recall

if __name__ == '__main__':


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = RUGD(root_dir='/home/bhanushali.r/Rellis-3D/Rellis-3D/', split='test', transform=transform)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(num_channels=3, num_classes=22)
    model.load_state_dict(torch.load('/home/bhanushali.r/Rellis-3D/Rellis-3D/rellis_3.0692608557785515_.pth', map_location=device))

    model.to(device)
    # Set the model to evaluation mode
    model.eval()


    
    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

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

            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()

            miou, pixel_acc, f1, recall = calculate_metrics(outputs, labels, 22)


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