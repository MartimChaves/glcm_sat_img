# imports
import numpy as np
from numpy.core.numeric import full
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops
# stratified k-folds
from sklearn.model_selection import train_test_split, StratifiedKFold
# from skmultilearn.model_selection.iterative_stratification import IterativeStratification
import matplotlib.pyplot as plt

# import torch
# from torchvision import transforms
# from PIL import Image
# import cv2

import argparse
import os
import copy
# from tqdm import tqdm

import time
import math
import csv

from img_utils import remove_collinear_features, high_corr_label, Image_Funcs


def parse_args():
    parser = argparse.ArgumentParser(description='Config')

    parser.add_argument('--root', type=str, default='./data/widsfixed/', help='Directory where files are.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random numbers.')

    # Data config
    parser.add_argument('--dataset_split', type=str, default='70,20,10', help='Percentage of dataset allocated to each set (train,val,test).')
    parser.add_argument('--rgb_imgs', type=str, default='True', help='If true, assume RGB images - else, assume gray images (TBA).')
    parser.add_argument('--glcm_only', type=str, default='True', help='If true, use only GLCM features.')

    # EDA
    parser.add_argument('--eda_rdm_img', type=str, default='True', help='If true, show random images (EDA).')
    parser.add_argument('--eda_rdm_img_num', type=int, default=5, help='Number of random images to show.')

    parser.add_argument('--eda_mean_img', type=str, default='True', help='If true, show class mean image (EDA).')
    parser.add_argument('--mean_img_counter', type=int, default=20, help='Number of imgs to use to calculate class mean.')

    parser.add_argument('--eda_eigen_img', type=str, default='True', help='If true, show eigen images (EDA).')
    parser.add_argument('--eigen_resize_factor', type=int, default=2, help='Resize factor for images used in eigen faces.')
    parser.add_argument('--eigen_img_counter', type=int, default=150, help='Number of imgs to use for eigen faces.')

    parser.add_argument('--eda_metrics_distr', type=str, default='True', help='If true, calculate distribution of features.')
    parser.add_argument('--stats_len', type=int, default=150, help='Resize factor for images used in eigen faces.')

    args = parser.parse_args()
    return args


# TO-DO: Create parent class that contains aug functions for EDA
class DS_aux(Image_Funcs):
    """
    Dataset-auxilliary class:
    Contains EDA and Dataset split
    """
    def __init__(self, args,
                 label_code={'No_OilPalm':0,
                             'Has_OilPalm':1}
                 ):
        super().__init__(args, label_code)
        self.data_split()

    def data_split(self):

        print("Calculating split fractions...")
        # get split fractions
        split_vals = self.args.dataset_split.split(",")
        trainset_fraction = int(split_vals[0])/100.
        valset_fraction = int(split_vals[1])/100.
        testset_fraction = int(split_vals[2])/100.
   
        # cross-validation
        self.num_folds = int((trainset_fraction+valset_fraction)/valset_fraction)
        self.folds = {}
        """
        folds (type: dict)
        ├───fold_1 (type: dict)
        │   ├───train (type: list)
        │   │   └───[train_img_id_1,train_img_id_2,...] # img_id = 'a849dhfhg.jpg'
        │   └───val (type: list)
        │       └───[val_img_id_1,val_img_id_2,...]
        ├───fold_2 (type: dict - same structure as fold_1)
        ...
        └───fold_n (last fold)
        """
        print("Generating folds...") # get data
        labels_df = pd.read_csv(self.labels_dir)
        imgs = np.array(labels_df['image'])
        labels = np.array(labels_df['label'])
        imgs, labels = shuffle(imgs, labels, random_state=self.args.seed)

        imgs_train_val, self.imgs_test, labels_train_val,\
        self.test_labels = train_test_split(imgs, labels, test_size=testset_fraction,
                                            random_state=self.args.seed, stratify=labels)

        valset_fraction = round(valset_fraction/(1-testset_fraction),2) # adapted after testset removal

        skf = StratifiedKFold(n_splits=self.num_folds)

        for i, (train_indexes, val_indexes) in enumerate(skf.split(imgs_train_val, labels_train_val)):
            self.folds["fold_"+str(i+1)] = {'train':imgs_train_val[train_indexes],
                                            'val':imgs_train_val[val_indexes],
                                            'train_labels':labels_train_val[train_indexes],
                                            'val_labels':labels_train_val[val_indexes]}

    # exploratory data analysis
    def show_rdm_imgs(self):

        def plot_img(num_imgs,i,img,counter,text):
            plt.subplot(3,num_imgs,i, xticks = [], yticks = [])
            plt.imshow(img)
            plt.title(f"{text} {counter}")

        num_imgs = self.args.eda_rdm_img_num

        train_imgs = np.random.choice(self.folds['fold_1']['train'],num_imgs,replace=False)
        val_imgs = np.random.choice(self.folds['fold_1']['val'],num_imgs,replace=False)
        test_imgs = np.random.choice(self.imgs_test,num_imgs,replace=False)

        train_imgs_file = [os.path.join(self.imgs_dir,img_file) for img_file in train_imgs]
        val_imgs_file = [os.path.join(self.imgs_dir,img_file) for img_file in val_imgs]
        test_imgs_file = [os.path.join(self.imgs_dir,img_file) for img_file in test_imgs]

        plt.figure(figsize = (8,6))

        for i in range(1,(num_imgs*3)+1):
            if i <= num_imgs:
                img = plt.imread(train_imgs_file[i-1])
                plot_img(num_imgs,i,img,i,"Train image")
            elif i>num_imgs and i <= num_imgs*2:
                j = i-num_imgs
                img = plt.imread(val_imgs_file[j-1])
                plot_img(num_imgs,i,img,j,"Val image")
            elif i > num_imgs*2:
                k = i-num_imgs*2
                img = plt.imread(test_imgs_file[k-1])
                plot_img(num_imgs,i,img,k,"Test image")

        #plt.show()
        plt.savefig("./plots/imgsexamples.png",dpi=200)
        plt.clf()
        
        # show random per class images
        labels = self.folds['fold_1']['train_labels']
        imgs = self.folds['fold_1']['train']
        
        for lbl_clss, lbl_indx in self.label_code.items():
            clss_imgs = self.get_class_imgs(lbl_indx,imgs,labels)
            clss_rdm_imgs = np.random.choice(clss_imgs,num_imgs,replace=False)
            
            if self.args.rgb_imgs == "True":
                self.plt_rgb_channels(clss_rdm_imgs, lbl_clss, num_imgs)
            else:
                self.plt_gray(clss_rdm_imgs, lbl_clss, num_imgs)
    
    def metrics_distr(self): # show metrics distributions
        
        labels = self.folds['fold_1']['train_labels']
        imgs = self.folds['fold_1']['train']
        
        full_stats_df = pd.DataFrame({})
        # calculate mean image for all classes
        for lbl_clss, lbl_indx in self.label_code.items():
            clss_imgs = self.get_class_imgs(lbl_indx,imgs,labels)
            class_feats = self.get_stats(clss_imgs, lbl_clss)
            
            class_feats_df_raw = (pd.io.json.json_normalize(class_feats, sep='_')) #pandas.json_normalize
            class_feats_df = pd.DataFrame({
                }).assign(**{col_name:np.concatenate(class_feats_df_raw[col_name].values)
                            for col_name in class_feats_df_raw.columns.tolist()})
            try:                
                len_column = len(class_feats['r']['min'])
            except:
                len_column = len(class_feats['r']['homogeneity'])
            
            label_arr = np.ones(len_column).astype(np.uint8)*self.label_code[lbl_clss]
            class_feats_df = class_feats_df.assign(label=label_arr)
            
            full_stats_df = pd.concat([full_stats_df, class_feats_df], axis=0, ignore_index=True)
        
        columns_of_interest = [colname for colname in full_stats_df.columns if colname != 'label']
        print("Preparing histogram plot...")
        sns.set(style='whitegrid', palette="deep", font_scale=0.8, rc={"figure.figsize": [8, 5]})
        full_stats_df[columns_of_interest].hist(bins=15, figsize=(20, 18), layout=(7, 8))
        plt.savefig('./plots/hist.png', dpi=200)
        plt.clf()
        print("Histogram plot saved.")
        
        # Are there highly correlated features?
        full_stats_df, high_corr_feats = remove_collinear_features(full_stats_df)
        for key in high_corr_feats: 
            print(f"{key}|{high_corr_feats[key]}")
        
        high_corr_df = pd.DataFrame.from_dict(high_corr_feats, orient='index')
        high_corr_df.to_csv("./plots/high_corr_feats.csv")
        
        sqrt_len_cols = np.sqrt(len(full_stats_df.columns))
        dec_part_sqrt = sqrt_len_cols % 1
        
        if dec_part_sqrt >= 0.5:
            dim_x = round(sqrt_len_cols) + 1
            dim_y = round(sqrt_len_cols) + 1
        elif dec_part_sqrt == 0.0:
            dim_x = round(sqrt_len_cols)
            dim_y = round(sqrt_len_cols)
        else:
            dim_x = round(sqrt_len_cols)
            dim_y = round(sqrt_len_cols) + 1
            
        fig, axes = plt.subplots(dim_x, dim_y, figsize=(28,26))
        
        count_x = 0
        count_y = 0
        for col in full_stats_df.columns:
            
            sns.kdeplot(ax=axes[count_x,count_y], data=full_stats_df,
                        hue='label', x=col, fill=True)
            
            if count_y < (dim_y-1):
                count_y += 1
            else:
                count_y = 0
                count_x += 1
            
        plt.savefig("./plots/kde.png", dpi=200)    
        plt.clf()
        
        print("Calculating features' correlation to label...")
        label_corr = {}
        label_thresh = 0.0
        feat_corr, val_corr, label_corr = high_corr_label(full_stats_df, threshold = label_thresh)
        
        label_corr_df = pd.DataFrame.from_dict(label_corr,orient='index')
        label_corr_df.to_csv("./plots/label_corr.csv")
        
        # get df with only the first five columns
        feat_list = list(feat_corr[-5::])
        feat_list.append('label')
        high_label_corr_df = full_stats_df[feat_list]
        
        sns.set_style("whitegrid")
        sns.pairplot(high_label_corr_df, hue="label")
        plt.savefig('./plots/pairplot.png', dpi=200)
        plt.clf() 
    
    def mean_image(self):
        
        labels = self.folds['fold_1']['train_labels']
        imgs = self.folds['fold_1']['train']
        
        # calculate mean image for all classes
        clss_mean_imgs = {}
        for i, lbl_clss in enumerate(self.label_code):
            indx = self.label_code[lbl_clss]
            mean_img = self.class_mean_img(indx, imgs, labels)
            
            clss_mean_imgs[lbl_clss] = mean_img
            
            plt.subplot(2,math.ceil(self.num_classes/2),i+1, xticks = [], yticks = [])
            plt.imshow(mean_img)
            plt.title(f"Class: {lbl_clss}")
            
        #plt.show()
        plt.savefig("./plots/meanimg.png",dpi=200)
        plt.clf()
        
        counter_img = 1
        for clss in clss_mean_imgs:
            if clss == 'No_OilPalm':
                continue
            else:
                clss_mean_imgs[clss] -= clss_mean_imgs['No_OilPalm']
                
                clss_mean_imgs[clss] += np.min(clss_mean_imgs[clss])
                clss_mean_imgs[clss] = np.divide(clss_mean_imgs[clss],np.max(clss_mean_imgs[clss]))
                clss_mean_imgs[clss] = np.multiply(clss_mean_imgs[clss],255)
                clss_mean_imgs[clss] = clss_mean_imgs[clss].astype(np.uint8)
                
                plt.subplot(2,math.ceil(self.num_classes/2),counter_img, xticks = [], yticks = [])
                plt.imshow(clss_mean_imgs[clss])
                plt.title(f"Class: {clss} - No_OilPalm")
                counter_img += 1
        
        #plt.show()
        plt.savefig("./plots/meanimgsubtract.png",dpi=200)
        plt.clf()
    
    def eigen_imgs(self):
        labels = self.folds['fold_1']['train_labels']
        imgs = self.folds['fold_1']['train']
        
        # calculate mean image for all classes
        for lbl_clss, lbl_indx in self.label_code.items():
            clss_imgs = self.get_class_imgs(lbl_indx,imgs,labels)
            full_mat = self.img2np(self.imgs_dir, clss_imgs)
            pca = self.class_eigen_imgs(full_mat)
            self.plot_pca(pca, lbl_clss)

def main(args):
    
    np.random.seed(args.seed)
        
    dataset_info = DS_aux(args)
    
    # plot random images
    if args.eda_rdm_img == "True":
        dataset_info.show_rdm_imgs()
        
    # average image for each class
    if args.eda_mean_img == "True":
        dataset_info.mean_image()

    # eigenimages
    if args.eda_eigen_img == "True":
        dataset_info.eigen_imgs()

    # feature distribution
    if args.eda_metrics_distr == "True":
        dataset_info.metrics_distr()
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    
    