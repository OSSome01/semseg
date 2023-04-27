import os
import shutil

# specify the paths to the .lst files and the root directory of the image data
train_lst_path = 'train.lst'
test_lst_path = 'test.lst'
val_lst_path = 'val.lst'
data_root = '/home/bhanushali.r/Rellis-3D/Rellis-3D/'

# create the directories for the train, test, and validation sets
train_dir = os.path.join(data_root, 'train')
test_dir = os.path.join(data_root, 'test')
val_dir = os.path.join(data_root, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# create the directories for the images and labels in the train, test, and validation sets
train_img_dir = os.path.join(train_dir, 'images')
train_lbl_dir = os.path.join(train_dir, 'labels')
test_img_dir = os.path.join(test_dir, 'images')
test_lbl_dir = os.path.join(test_dir, 'labels')
val_img_dir = os.path.join(val_dir, 'images')
val_lbl_dir = os.path.join(val_dir, 'labels')
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_lbl_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# read the .lst files and move the corresponding image files to the appropriate directories
with open(train_lst_path, 'r') as f:
    train_paths = [line.strip().split() for line in f.readlines()]
for path, label_path in train_paths:
    src_path = os.path.join(data_root, path)
    dst_img_path = os.path.join(train_img_dir, os.path.basename(path))
    dst_lbl_path = os.path.join(train_lbl_dir, os.path.basename(label_path))
    shutil.copy(src_path, dst_img_path)
    shutil.copy(label_path, dst_lbl_path)

with open(test_lst_path, 'r') as f:
    test_paths = [line.strip().split() for line in f.readlines()]
for path, label_path in test_paths:
    src_path = os.path.join(data_root, path)
    dst_img_path = os.path.join(test_img_dir, os.path.basename(path))
    dst_lbl_path = os.path.join(test_lbl_dir, os.path.basename(label_path))
    shutil.copy(src_path, dst_img_path)
    shutil.copy(label_path, dst_lbl_path)

with open(val_lst_path, 'r') as f:
    val_paths = [line.strip().split() for line in f.readlines()]
for path, label_path in val_paths:
    src_path = os.path.join(data_root, path)
    dst_img_path = os.path.join(val_img_dir, os.path.basename(path))
    dst_lbl_path = os.path.join(val_lbl_dir, os.path.basename(label_path))
    shutil.copy(src_path, dst_img_path)
    shutil.copy(label_path, dst_lbl_path)