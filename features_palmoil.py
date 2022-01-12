# imports
import numpy as np
from numpy.core.numeric import full
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops
# stratified k-folds
from skmultilearn.model_selection.iterative_stratification import IterativeStratification
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from PIL import Image
import cv2

import argparse
import os
import copy
from tqdm import tqdm

import time
import math
import csv


def parse_args():
    parser = argparse.ArgumentParser(description='Config')
    
    parser.add_argument('--root', type=str, default='./data/palmoil/', help='Directory where files are.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random numbers.')
    
    # Data config
    parser.add_argument('--dataset_split', type=str, default='70,20,10', help='Percentage of dataset allocated to each set (train,val,test).')
    parser.add_argument('--rgb_imgs', type=str, default='False', help='If true, assume RGB images - else, assume gray images.')
    
    # EDA
    parser.add_argument('--eda', type=str, default='True', help='If true, carry out exploratory data analysis (EDA).')
    parser.add_argument('--eda_rdm_img', type=str, default='True', help='If true, show random images (EDA).')
    parser.add_argument('--eda_rdm_img_num', type=int, default=5, help='Number of random images to show.')
    
    parser.add_argument('--eda_mean_img', type=str, default='False', help='If true, show class mean image (EDA).')
    parser.add_argument('--mean_img_counter', type=int, default=30, help='Number of imgs to use to calculate class mean.')
    
    parser.add_argument('--eda_eigen_img', type=str, default='False', help='If true, show eigen images (EDA).')
    parser.add_argument('--eigen_resize_factor', type=int, default=2, help='Resize factor for images used in eigen faces.')
    parser.add_argument('--eigen_img_counter', type=int, default=300, help='Number of imgs to use for eigen faces.')
    
    parser.add_argument('--stats_len', type=int, default=5, help='Resize factor for images used in eigen faces.')
    args = parser.parse_args()
    return args

def main(args):
    
    np.random.seed(args.seed)
    
    #use cuda 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # dataset generator params
    params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 2}
    
    dataset_info = DS_aux(args)
    
    if args.eda == "True":
        dataset_info.eda()
    
    """
    for i in range(1,dataset_info.num_folds+1):
        # TO-DO: turn this into a function
        train_imgs = dataset_info.folds["fold_"+str(i)]['train']
        train_labels = dataset_info.folds["fold_"+str(i)]['train_labels']
        train_set = Dataset(args,train_imgs,train_labels)
        train_gen = torch.utils.data.DataLoader(train_set, **params)
        
        val_imgs = dataset_info.folds["fold_"+str(i)]['val']
        val_labels = dataset_info.folds["fold_"+str(i)]['val_labels']
        val_set = Dataset(args,val_imgs,val_labels)
        val_gen = torch.utils.data.DataLoader(val_set, **params)
        
        for epoch in range(1):
            print(f"Training - epoch: {epoch}")
            for imgs, labels, idxs in tqdm(train_gen):
                imgs, labels = imgs.to(device), labels.to(device) 
            
            print(f"Validating - epoch: {epoch}")
            for imgs, labels, idxs in tqdm(val_gen):
                imgs, labels = imgs.to(device), labels.to(device) 
    """
if __name__ == "__main__":
    #args = parse_args()
    #main(args)
    
    path = "./data/widsdatathon2019/"
    
    labels = os.path.join(path,'traininglabels.csv')
    train = os.path.join(path,'train_images/')
    
    # image path + 2017
    
    