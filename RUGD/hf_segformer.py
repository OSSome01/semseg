import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
import wandb


# wandb.init(project='sem_seg_segformer', 
#             entity='ossome',
#             config={'learning_rate':0.00006,
#                     'epochs':100})

class RUGD(Dataset):
    def __init__(self, root_dir, feature_extractor, color_map, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.color_map = color_map

        # Load the list of image and label filenames

        self.image_filenames = sorted(glob.glob(os.path.join(self.root_dir, '*.png')))
        self.label_filenames = sorted(glob.glob(os.path.join(self.root_dir+'_labels', '*.png')))

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
        label_size = np.array(label).shape
        label_classes = np.zeros((label_size[0]*label_size[1]), dtype=np.uint8)
        label = np.array(label)
        label_pixels = label.reshape(-1,3)
        for i in range(label_pixels.shape[0]):
          rgb = tuple(label_pixels[i])
          if rgb in self.color_map:
            label_classes[i] = self.color_map[rgb]
        label = label_classes.reshape(label_size[0],label_size[1])    
        
        
        encoded_inputs = self.feature_extractor(image, Image.fromarray(label), return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()
        return encoded_inputs


if __name__ == '__main__':
    color_map = pd.read_csv('RUGD_annotation-colormap.txt', sep=" ", header=None)
    color_map.columns = ["label_idx", "label", "R", "G", "B"]
    color_map.head()

    label2id = {label: id for id, label in enumerate(color_map.label)}
    id2label = {id: label for id, label in enumerate(color_map.label)}
    id2color = {id: [R,G,B] for id, (R,G,B) in enumerate(zip(color_map.R, color_map.G, color_map.B))}
    color2id = {(R,G,B): id for id, (R,G,B) in enumerate(zip(color_map.R, color_map.G, color_map.B))}
    # print(color2id)

    del id2color[0]
    id2color = {id-1: color for id, color in id2color.items()}
    print(id2color)
    del id2label[0]
    label2id = {label: id-1 for id, label in id2label.items()}
    id2label = {id-1: label for id, label in id2label.items()}
    print(id2label)

    train_dir = '/home/ossome/semantic_segmentation/sem_seg/RUGD_small/train'
    val_dir = '/home/ossome/semantic_segmentation/sem_seg/RUGD_small/val'
    feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

    train_dataset = RUGD(root_dir=train_dir, feature_extractor=feature_extractor, color_map=color2id)
    val_dataset = RUGD(root_dir=val_dir, feature_extractor=feature_extractor, color_map=color2id)

    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_metric = load_metric("mean_iou")
    val_metric = load_metric("mean_iou")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)

    num_epochs = 15
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for idx, batch in enumerate(tqdm(trainloader)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            train_loss, logits = outputs.loss, outputs.logits
            
            train_loss.backward()
            optimizer.step()
            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)
                train_metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
        train_metrics = train_metric.compute(num_labels=len(id2label), ignore_index=255,reduce_labels=False)
        print("Loss:", train_loss.item())
        print("Mean_iou:", train_metrics["mean_iou"])
        print("Mean accuracy:", train_metrics["mean_accuracy"])

        # Test the model on the val set
        model.eval()
        val_loss = 0.0
        for idx, batch in enumerate((valloader)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss, logits = outputs.loss, outputs.logits

                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)
                
                val_metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())    
        val_metrics = val_metric.compute(num_labels=len(id2label), ignore_index=255, reduce_labels=False)
        print("Val Loss:", val_loss.item())
        print("Val Mean_iou:", val_metrics["mean_iou"])
        print("Val Mean accuracy:", val_metrics["mean_accuracy"])
                
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'segformer_best_model_weights.pth')
        # wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        # scheduler.step()

        # Print the loss for the current epoch
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')



