# imports
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
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

def parse_args():
    parser = argparse.ArgumentParser(description='Config')
    
    parser.add_argument('--root', type=str, default='./data/apl_trees/plant-pathology-2021-fgvc8/', help='Directory where files are.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random numbers.')
    
    # Data config
    parser.add_argument('--dataset_split', type=str, default='70,20,10', help='Percentage of dataset allocated to each set (train,val,test).')
    
    # EDA
    parser.add_argument('--eda', type=str, default='True', help='If true, carry out exploratory data analysis (EDA).')
    parser.add_argument('--eda_rdm_img', type=str, default='False', help='If true, show random images (EDA).')
    parser.add_argument('--eda_rdm_img_num', type=int, default=5, help='Number of random images to show.')
    
    parser.add_argument('--eda_mean_img', type=str, default='True', help='If true, show class mean image (EDA).')
    parser.add_argument('--mean_img_counter', type=int, default=30, help='Number of imgs to use to calculate class mean.')
    
    parser.add_argument('--eda_eigen_img', type=str, default='True', help='If true, show eigen images (EDA).')
    parser.add_argument('--eigen_resize_factor', type=int, default=8, help='Resize factor for images used in eigen faces.')
    parser.add_argument('--eigen_img_counter', type=int, default=30, help='Number of imgs to use for eigen faces.')
    
    args = parser.parse_args()
    return args


class Image_Funcs():
    
    def __init__(self, args, label_code):
        
        self.args = args
        self.imgs_dir = os.path.join(self.args.root,"train_images")
        self.labels_dir = os.path.join(self.args.root,"train.csv")
        #self.extra_imgs_dir = os.path.join(self.args.root,"test_images")
        self.label_code = label_code
        self.num_classes = len(label_code.keys())
        
        return
    
    def norm_uint8(self, img): # TO-DO
        img = np.add(img, np.min(img))
        return img
    
    def img2np(self, path, list_of_filename, size = (2672, 4000, 3)):
        
        # init full list of images array
        res_fac = self.args.eigen_resize_factor
        a_s = (int(size[0]/res_fac),int(size[1]/res_fac)) # adjusted_size 334,500
        full_mat = np.zeros((self.args.eigen_img_counter,a_s[0]*a_s[1]*size[2]))
        
        # iterating through each file
        for idx, fn in enumerate(tqdm(list_of_filename)):
            fp = os.path.join(path,fn)
            # read image
            img = plt.imread(fp)
            # resize image
            img = cv2.resize(img, dsize=(a_s[1],a_s[0]), interpolation=cv2.INTER_CUBIC)
            # turn that into a vector / 1D array
            img = img.ravel()
            full_mat[idx] = img
            
            if idx >= self.args.eigen_img_counter-1:
                break
            
        return full_mat
    
    def class_eigen_imgs(self, full_mat, n_comp = 0.7):
        
        # fit PCA to describe n_comp * variability in the class
        pca = PCA(n_components = n_comp, whiten = True)
        pca.fit(full_mat)
        print(f'Number of PC: {pca.n_components_}')
        return pca
    
    def plot_pca(self, pca, lbl_clss, size = (334, 500, 3)):
        # plot eigenimages in a grid
        n = pca.n_components_
        fig = plt.figure(figsize=(8, 8))
        r = int(n**.5)
        c = math.ceil(n/ r)
        
        for i in range(n):
            ax = fig.add_subplot(r, c, i + 1, xticks = [], yticks = [])
            pca_component = pca.components_[i]
            pca_component = pca.components_[i]+np.min(pca.components_[i])
            pca_component /= np.max(pca_component)
            pca_component *= 255
            pca_component = pca_component.astype(np.uint8) 
            ax.imshow(pca_component.reshape(size), cmap='Greys_r')
            
        plt.axis('off')
        plt.title(f'Class: {lbl_clss}')
        plt.show()

# TO-DO: Create parent class that contains aug functions for EDA
class DS_aux(Image_Funcs):
    """
    Dataset-auxilliary class:
    Contains EDA and Dataset split
    """
    def __init__(self, args,
                 label_code={'healthy':0,
                             'scab':1,
                             'frog_eye_leaf_spot':2,
                             'complex':3,
                             'rust':4,
                             'powdery_mildew':5}
                 ):
        super().__init__(args, label_code)
        
        
        self.data_split()
        
        return

    def convert_label(self, lbl_raw):
        onehot = [0 for _ in range(len(self.label_code.keys()))]
        
        if not type(lbl_raw)==str:
            raise ValueError(F'Label not string, label type: {type(lbl_raw)}')
        
        if ' ' not in lbl_raw:
            idx = self.label_code[lbl_raw]
            onehot[idx] = 1
        elif ' ' in lbl_raw:
            split_lbl = lbl_raw.split(' ')
            for single_lbl in split_lbl:
                idx = self.label_code[single_lbl]
                onehot[idx] = 1
        else:
            pass
        
        return onehot
    
    def data_split(self):
        
        print("Getting datafiles paths...")
        img_files = os.listdir(self.imgs_dir)
        np.random.shuffle(img_files)
        
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
        onehot_labels = np.array([self.convert_label(lbl) for lbl in labels_df['labels']])
        imgs, onehot_labels = shuffle(imgs, onehot_labels, random_state=self.args.seed)
        
        ## order means the order-th label combinations (in this case it's 3- maximum number of concurrent labels is 3)
        """
        lens = []
        for lbl in a:
            if ' ' in lbl:
                x = lbl.split(' ')
                lens.append(len(x))
        order = np.max(lens)
        """
        stratifier_test = IterativeStratification(n_splits=2, order=3, sample_distribution_per_fold=[testset_fraction, 1-testset_fraction])
        train_val_indexes, test_indexes = next(stratifier_test.split(imgs, onehot_labels))
        self.imgs_test, self.labels_test = imgs[test_indexes], onehot_labels[test_indexes]
        
        imgs_train_val, oneh_train_val = imgs[train_val_indexes], onehot_labels[train_val_indexes]
        valset_fraction = round(valset_fraction/(1-testset_fraction),2) # adapted after testset removal
        
        # get stratifier
        sample_distr = [valset_fraction for _ in range(self.num_folds)]
        sample_distr.append(1-np.sum(sample_distr))
        stratifier_val = IterativeStratification(n_splits=(self.num_folds+1), order=3, sample_distribution_per_fold=sample_distr)
        # generate folds
        for i in tqdm(range(self.num_folds)):
            train_indexes, val_indexes = next(stratifier_val.split(imgs_train_val, oneh_train_val))
            self.folds["fold_"+str(i+1)] = {'train':imgs_train_val[train_indexes],
                                            'val':imgs_train_val[val_indexes],
                                            'train_labels':oneh_train_val[train_indexes],
                                            'val_labels':oneh_train_val[val_indexes]}
            
        """ About 25% of intersection between validation sets of different folds (not great, not terrible)
        intersect = np.intersect1d(self.folds["fold_1"]['val'],self.folds["fold_2"]['val']).shape[0]
        fold_size = self.folds["fold_1"]['val'].shape[0]
        print("Percentage of intersection between different folds val sets: ",round(intersect/fold_size*100,2)) 
        """
        return
    
    ### exploratory data analysis
    def show_rdm_imgs(self):

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
    
    def calc_mean_img(self, path, list_of_filename, size = (2672, 4000, 3)):
        # iterating through each file
        flag = False
        counter = 0
        for idx, fn in enumerate(tqdm(list_of_filename)):
            fp = os.path.join(path,fn)
            # read image (array)
            img = plt.imread(fp)
            if img.shape == size: #TO-DO: Improve this (add functionality)
                # turn that into a vector / 1D array
                img = img.ravel()
                img = img.astype(np.float64)
                if flag:
                    full_mat = np.add(full_mat,img)
                    counter += 1
                else:
                    full_mat = img
                    flag = True
                    counter += 1
            else:
                continue
            if idx > self.args.mean_img_counter:
                break
            
        full_mat = np.divide(full_mat,counter)
        mean_img = full_mat.reshape(size)
        mean_img = mean_img.astype(np.uint8)
        return mean_img
    
    def get_class_imgs(self,indx,imgs,labels):
        onehot_lbl = np.zeros(len(self.label_code.keys()))
        onehot_lbl[indx] = 1
        
        imgs_indx = [idx for idx,lbl in enumerate(labels) if (lbl == onehot_lbl).all()]
        class_imgs = imgs[imgs_indx]
        
        return class_imgs
    
    def class_mean_img(self, indx, imgs, labels):
        
        class_imgs = self.get_class_imgs(indx,imgs,labels)
        class_mean_img = self.calc_mean_img(self.imgs_dir,class_imgs)
        
        return class_mean_img
    
    def mean_image(self):
        
        labels = self.folds['fold_1']['train_labels']
        imgs = self.folds['fold_1']['train']
        
        # calculate mean image for all classes
        clss_mean_imgs = {}
        for i, lbl_clss in enumerate(self.label_code):
            indx = self.label_code[lbl_clss]
            mean_img = self.class_mean_img(indx, imgs, labels)
            
            clss_mean_imgs[lbl_clss] = mean_img
            
            plt.subplot(2,math.ceil(self.num_classes/2),i+1)
            plt.imshow(mean_img)
            plt.title(f"Class: {lbl_clss}")
            
        plt.show()
        
        counter_img = 1
        for clss in clss_mean_imgs:
            if clss == 'healthy':
                continue
            else:
                clss_mean_imgs[clss] -= clss_mean_imgs['healthy']
                plt.subplot(2,math.ceil(self.num_classes/2),counter_img)
                plt.imshow(clss_mean_imgs[clss])
                plt.title(f"Class: {clss} (- healthy))")
                counter_img += 1
        
        plt.show()
        return
    
    def eigen_imgs(self):
        labels = self.folds['fold_1']['train_labels']
        imgs = self.folds['fold_1']['train']
        
        # calculate mean image for all classes
        for i, lbl_clss in enumerate(self.label_code):
            indx = self.label_code[lbl_clss]
            clss_imgs = self.get_class_imgs(indx,imgs,labels)
            full_mat = self.img2np(self.imgs_dir, clss_imgs)
            pca = self.class_eigen_imgs(full_mat)
            self.plot_pca(pca, lbl_clss)
        
        return
    
    def eda(self):

        # random images
        if self.args.eda_rdm_img == "True":
            self.show_rdm_imgs()
            
        # average image for each class
        if self.args.eda_mean_img == "True":
            self.mean_image()

        # eigenfaces
        if self.args.eda_eigen_img == "True":
            self.eigen_imgs()

        # RGB/HSV/Grey
        # Max value, min value, mean, std for each class
        # correlation for each class and in general

        return

### feature development

### feature selection

### Dataset class
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, args, data, labels):
        
        self.args = args
        self.data = data
        self.labels = labels
        
        self.imgs_dir = os.path.join(self.args.root,"train_images")
        
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        
        img_file = self.data[idx]
        img_path = os.path.join(self.imgs_dir,img_file)
        img = plt.imread(img_path)
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        
        label = self.labels[idx]
        
        return img, label, idx


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
        
        # TO-DO: transform arrays to tensors and pass to GPU
        for epoch in range(1):
            print(f"Training - epoch: {epoch}")
            for imgs, labels, idxs in tqdm(train_gen):
                imgs, labels = imgs.to(device), labels.to(device) 
            
            print(f"Validating - epoch: {epoch}")
            for imgs, labels, idxs in tqdm(val_gen):
                imgs, labels = imgs.to(device), labels.to(device) 
    
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)