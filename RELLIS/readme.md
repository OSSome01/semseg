Data Preparation file - rellis_data.py
Vanilla U-Net training - final_project.py
Attention U-Net training - final_project_attention.py

------------------------------------------------------
Training on the data
------------------------------------------------------
1. Download Rellis-3D data and form the correct file structure as mentioned on the website - https://github.com/unmannedlab/RELLIS-3D
2. Edit the path in rellis_data.py file and run it. You should have three folders - train,test,val
3. Open the Vanilla U-Net and Attention U-Net training file and change the path for the root directory.
4. Run the respective file for training.
------------------------------------------------------
Evaluation and Inference
------------------------------------------------------
1. Run the evaluation_metrics.py after changing the root directory in it file to get the mIoU, Pixel accuracy, F1 Score and Recall
2. Run the infer.py after changing the directories to get the generated mask for the image