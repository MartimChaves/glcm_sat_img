# imports
import numpy as np
import matplotlib.pyplot as plt
import torch

import argparse
import os
import copy
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Config')
    
    parser.add_argument('--root', type=str, default='./data/apl_trees/plant-pathology-2021-fgvc8/', help='Directory where files are.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random numbers.')
    
    # Data config
    parser.add_argument('--dataset_split', type=str, default='70,20,10', help='Percentage of dataset allocated to each set (train,val,test).')
    
    # EDA
    parser.add_argument('--eda', type=str, default='True', help='If true, carry out exploratory data analysis (EDA).')
    parser.add_argument('--eda_rdm_img', type=str, default='True', help='If true, show random images (EDA).')
    parser.add_argument('--eda_rdm_img_num', type=int, default=5, help='Number of random images to show.')
    
    args = parser.parse_args()
    return args

class DS_aux():
    """
    Dataset-auxilliary class:
    Contains EDA and Dataset split
    """
    
    
    def __init__(self,args):
        
        self.args = args
        self.imgs_dir = os.path.join(self.args.root,"train_images")
        #self.extra_imgs_dir = os.path.join(self.args.root,"test_images")
        
        self.data_split()
        
        return

    def data_split(self):
        
        img_files = os.listdir(self.imgs_dir)
        np.random.shuffle(img_files)
        
        # get split fractions
        trainset_fraction = int(self.args.dataset_split.split(",")[0])/100.
        valset_fraction = int(self.args.dataset_split.split(",")[1])/100.
        #testset_fraction = int(self.args.dataset_split.split(",")[2])/100.
        
        trainset_len = int(len(img_files)*trainset_fraction)
        valset_len = int(len(img_files)*valset_fraction)
        testset_len = len(img_files) - trainset_len - valset_len # prevent rounding errors
        
        self.testset_files = img_files[-testset_len::]
        img_files = img_files[0:-testset_len]
        
        # cross-validation
        k_num = int((trainset_fraction+valset_fraction)/valset_fraction)
        
        self.folds = {}
        for i in range(k_num):
            val_files = img_files[i*valset_len:(i+1)*valset_len]
            train_files = [file for file in img_files if file not in val_files]
            self.folds["fold_"+str(i+1)] = {'train':train_files,'val':val_files}
            
        return
    
    ### exploratory data analysis
    def show_rdm_imgs(self):

        num_imgs = self.args.eda_rdm_img_num
        
        train_imgs = np.random.choice(self.folds['fold_1']['train'],num_imgs,replace=False)
        val_imgs = np.random.choice(self.folds['fold_1']['val'],num_imgs,replace=False)
        test_imgs = np.random.choice(self.testset_files,num_imgs,replace=False)
        
        train_imgs_file = [os.path.join(self.imgs_dir,img_file) for img_file in train_imgs]
        val_imgs_file = [os.path.join(self.imgs_dir,img_file) for img_file in val_imgs]
        test_imgs_file = [os.path.join(self.imgs_dir,img_file) for img_file in test_imgs]
        
        plt.figure(figsize = (8,6))
        
        for i in range(1,(num_imgs*3)+1):
            if i <= num_imgs:
                img = plt.imread(train_imgs_file[i-1])
                plt.subplot(3,num_imgs,i)
                plt.imshow(img)
                plt.title("Train image " + str(i))
            elif i>num_imgs and i <= num_imgs*2:
                j = i-num_imgs
                img = plt.imread(val_imgs_file[j-1])
                plt.subplot(3,num_imgs,i)
                plt.imshow(img)
                plt.title("Val image " + str(j))
            elif i > num_imgs*2:
                k = i-num_imgs*2
                img = plt.imread(test_imgs_file[k-1])
                plt.subplot(3,num_imgs,i)
                plt.imshow(img)
                plt.title("Test image " + str(k))
        
        plt.show()
        
        return

    def eda(self):

        # random images
        if self.args.eda_rdm_img == "True":
            self.show_rdm_imgs()
            
        # average image for each class
        # ravel image into vector
        # reshape to original size

        # difference between average of images

        # variability 

        # eigenfaces

        # RGB/HSV/Grey
        # Max value, min value, mean, std for each class
        # correlation for each class and in general

        return

### feature development

### feature selection

### Dataset class
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self):
        
        return
    
    def __len__(self):
        return 0
    
    def __getitem__(self,idx):
        return idx


def main(args):
    
    np.random.seed(args.seed)
    
    dataset_info = DS_aux(args)
    
    if args.eda == "True":
        dataset_info.eda()
    
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)