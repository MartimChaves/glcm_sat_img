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
    
    parser.add_argument('--eda_mean_img', type=str, default='False', help='If true, show class mean image (EDA).')
    parser.add_argument('--mean_img_counter', type=int, default=30, help='Number of imgs to use to calculate class mean.')
    
    parser.add_argument('--eda_eigen_img', type=str, default='False', help='If true, show eigen images (EDA).')
    parser.add_argument('--eigen_resize_factor', type=int, default=8, help='Resize factor for images used in eigen faces.')
    parser.add_argument('--eigen_img_counter', type=int, default=300, help='Number of imgs to use for eigen faces.')
    
    parser.add_argument('--stats_len', type=int, default=2, help='Resize factor for images used in eigen faces.')
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
    
    def norm_uint8(self, img):
        img = np.add(img, np.min(img))
        img = np.divide(img,np.max(img))
        img = np.multiply(img,255)
        img = img.astype(np.uint8) 
        return img
    
    def rgb2gray(self,img):
        
        img = np.copy(img.astype(np.float32))
        gray = np.add(img[0::,0::,0],np.add(img[0::,0::,1],img[0::,0::,2]))
        gray = np.divide(gray,3)
        gray = gray.astype(np.uint8)
        
        return gray
    
    def get_glcm(self, img, offsetdist=[1], offsetang = [7*np.pi/4], imgvals = 256):
        glcm = graycomatrix(img, distances=offsetdist, angles=offsetang, levels=imgvals,
                        symmetric=False, normed=True)
        return glcm
    
    def get_glcm_metrics(self, prop = '',glcm = ''):
        return graycoprops(glcm, prop)[0, 0]
        
    def get_stats(self, clss_imgs, lbl_clss):
        # img is a single channel
        init_base_stats = np.multiply(np.ones(self.args.stats_len),-1)
        stats = {
            'min'        : np.copy(init_base_stats),
            'max'        : np.copy(init_base_stats),
            'mean'       : np.copy(init_base_stats),
            'std'        : np.copy(init_base_stats),
            'homogeneity': np.copy(init_base_stats),
            'contrast'   : np.copy(init_base_stats),
            'energy'     : np.copy(init_base_stats),
            'correlation': np.copy(init_base_stats)
        }
        
        stats_calc = {
            'min'        : np.min,
            'max'        : np.max,
            'mean'       : np.mean,
            'std'        : np.std,
            'homogeneity': self.get_glcm_metrics,
            'contrast'   : self.get_glcm_metrics,
            'energy'     : self.get_glcm_metrics,
            'correlation': self.get_glcm_metrics
        }
        
        channel_stats = {
            'r'   : copy.deepcopy(stats),
            'g'   : copy.deepcopy(stats),
            'b'   : copy.deepcopy(stats),
            'h'   : copy.deepcopy(stats),
            's'   : copy.deepcopy(stats),
            'v'   : copy.deepcopy(stats),
            'gray': copy.deepcopy(stats)
        }
        
        channels = {0: 'r', 1: 'g', 2: 'b'}
        hsv = {0: 'h', 1: 's', 2: 'v'}
        for file in tqdm(clss_imgs):
            
            # get indx of unfilled stat
            available_indxs = np.where(channel_stats['r']['min']==-1)[0]
            if not available_indxs.any():
                break
            stats_indx = np.min(available_indxs)
            
            img_path = os.path.join(self.imgs_dir,file)
            img = plt.imread(img_path)
            # RGB
            for idx in range(img.shape[-1]):
                channel = img[..., idx]
                channel_glcm = self.get_glcm(channel)
                np_args = {'a':channel}
                obj_args = {'glcm':channel_glcm}
                for stat in stats:
                    try:
                        stat_val = stats_calc[stat](**np_args)
                    except:
                        stat_val = stats_calc[stat](prop = stat, **obj_args)
                    channel_stats[channels[idx]][stat][stats_indx] = stat_val
            # Gray
            gray_img = self.rgb2gray(img)
            gray_glcm = self.get_glcm(gray_img)
            for stat in stats:
                try:
                    stat_val = stats_calc[stat](**{'a':gray_img})
                except:
                    stat_val = stats_calc[stat](prop = stat, **{'glcm':gray_glcm})
                channel_stats['gray'][stat][stats_indx] = stat_val
            # HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            for idx in range(img_hsv.shape[-1]):
                channel = img_hsv[..., idx]
                channel_glcm = self.get_glcm(channel)
                np_args = {'a':channel}
                obj_args = {'glcm':channel_glcm}
                for stat in stats:
                    try:
                        stat_val = stats_calc[stat](**np_args)
                    except:
                        stat_val = stats_calc[stat](prop = stat, **obj_args)
                    channel_stats[hsv[idx]][stat][stats_indx] = stat_val
                    
        return channel_stats
    
    def plt_rgb_channels(self, clss_rdm_imgs, lbl_clss, num_imgs):
        
        def plot_img(num_imgs,i,img,text):
            plt.subplot(num_imgs,8,i)
            plt.imshow(img)
            plt.title(f"{text} {i}")
        
        plt_img_counter = 1
        channels = {0: 'R', 1: 'G', 2: 'B'}
        hsv = {0: 'H', 1: 'S', 2: 'V'}
        for file in clss_rdm_imgs:
            img_path = os.path.join(self.imgs_dir,file)
            img = plt.imread(img_path)
            
            plot_img(num_imgs,plt_img_counter,img,f"{lbl_clss} RBG: ")
            plt_img_counter += 1
            
            for idx in range(img.shape[-1]):
                channel = img[..., idx]
                plot_img(num_imgs, plt_img_counter, channel, f"{lbl_clss} {channels[idx]}: ")
                plt_img_counter += 1
                
            gray_img = self.rgb2gray(img)
            plot_img(num_imgs, plt_img_counter, gray_img, f"{lbl_clss} grayscale: ")
            plt_img_counter += 1
            
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            for idx in range(img_hsv.shape[-1]):
                channel = img_hsv[..., idx]
                plot_img(num_imgs, plt_img_counter, channel, f"{lbl_clss} {hsv[idx]}: ")
                plt_img_counter += 1

        plt.show()
        
    # *** MEAN IMAGE ***
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
            if idx > self.args.mean_img_counter-1:
                break
            
        full_mat = np.divide(full_mat,counter)
        mean_img = full_mat.reshape(size)
        mean_img = mean_img.astype(np.uint8)
        return mean_img
    
    def get_class_imgs(self,indx,imgs,labels):
        """
        Returns file names of images that belong only to one class,
        sort of returns 'pure' class representatives
        """
        onehot_lbl = np.zeros(len(self.label_code.keys()))
        onehot_lbl[indx] = 1
        
        imgs_indx = [idx for idx,lbl in enumerate(labels) if (lbl == onehot_lbl).all()]
        class_imgs = imgs[imgs_indx]
        
        return class_imgs
    
    def class_mean_img(self, indx, imgs, labels):
        
        class_imgs = self.get_class_imgs(indx,imgs,labels)
        class_mean_img = self.calc_mean_img(self.imgs_dir,class_imgs)
        
        return class_mean_img
    
    # *** EIGEN IMAGES **
    def img2np(self, path, list_of_filename, size = (2672, 4000, 3)):
        
        # init full list of images array
        res_fac = self.args.eigen_resize_factor
        a_s = (int(size[0]/res_fac),int(size[1]/res_fac)) # adjusted_size 334,500
        full_mat = np.zeros((self.args.eigen_img_counter,a_s[0]*a_s[1]))
        
        # iterating through each file
        real_counter = 0
        for idx, fn in enumerate(tqdm(list_of_filename)):
            fp = os.path.join(path,fn)
            # read image
            img = plt.imread(fp)
            
            if img.shape == size:
                img = self.rgb2gray(img)
                # resize image
                img = cv2.resize(img, dsize=(a_s[1],a_s[0]), interpolation=cv2.INTER_AREA) # INTER_CUBIC if rgb img
                # turn that into a vector / 1D array
                img = img.ravel()
                full_mat[real_counter] = img
                real_counter += 1
            
            if real_counter >= self.args.eigen_img_counter:
                break
            
        return full_mat
    
    def class_eigen_imgs(self, full_mat, n_comp = 0.7):
        
        # fit PCA to describe n_comp * variability in the class
        pca = PCA(n_components = n_comp, whiten = True)
        print('Fitting PCA...')
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
            ax.imshow(self.norm_uint8(pca.components_[i]).reshape(size[0],size[1]), cmap='Greys_r')
            
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
    
    ### exploratory data analysis
    def show_rdm_imgs(self):
        
        def plot_img(num_imgs,i,img,counter,text):
            plt.subplot(3,num_imgs,i)
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
        
        plt.show()
        
        # show random per class images
        labels = self.folds['fold_1']['train_labels']
        imgs = self.folds['fold_1']['train']
        
        # calculate mean image for all classes
        for lbl_clss, lbl_indx in self.label_code.items():
            clss_imgs = self.get_class_imgs(lbl_indx,imgs,labels)
            clss_rdm_imgs = np.random.choice(clss_imgs,num_imgs,replace=False)
            self.plt_rgb_channels(clss_rdm_imgs, lbl_clss, num_imgs)
            stats = self.get_stats(clss_imgs, lbl_clss)
            # RBG, separate channels, separate HSV, greyscale
    
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
                
            len_column = len(class_feats['r']['min'])
            label_arr = np.ones(len_column).astype(np.uint8)*self.label_code[lbl_clss]
            class_feats_df = class_feats_df.assign(label=label_arr)
            
            full_stats_df = pd.concat([full_stats_df, class_feats_df], axis=0)
        
        columns_of_interest = [colname for colname in full_stats_df.columns if colname != 'label']
        sns.set(style='whitegrid', palette="deep", font_scale=0.8, rc={"figure.figsize": [8, 5]})
        full_stats_df[columns_of_interest].hist(bins=15, figsize=(15, 6), layout=(7, 8))
                
        # max, min, mean, std
    
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
    
    def eigen_imgs(self):
        labels = self.folds['fold_1']['train_labels']
        imgs = self.folds['fold_1']['train']
        
        # calculate mean image for all classes
        for lbl_clss, lbl_indx in self.label_code.items():
            clss_imgs = self.get_class_imgs(lbl_indx,imgs,labels)
            full_mat = self.img2np(self.imgs_dir, clss_imgs)
            pca = self.class_eigen_imgs(full_mat)
            self.plot_pca(pca, lbl_clss)
        
    def eda(self):

        self.metrics_distr()
        
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

### feature development

### feature selection

### Dataset class
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, args, data, labels):
        
        self.args = args
        self.data = data
        self.labels = labels
        
        self.imgs_dir = os.path.join(self.args.root,"train_images")
    
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

if __name__ == "__main__":
    args = parse_args()
    main(args)