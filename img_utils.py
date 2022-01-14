import os
from tkinter import font
import numpy as np
import math
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops
import copy
from tqdm import tqdm


def remove_collinear_features(x, threshold = 0.95, drop_cols = True, verbose = False):
    '''
    Original Author: Synergix (Stackoverflow) - with sligh modifications
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    high_corr = {}
    
    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            if col[0]=='label' or row[0]=='label':
                continue
            
            # If correlation exceeds the threshold
            if val >= threshold:
                corr_val = round(val[0][0],2)
                # Print the correlated features and the correlation value
                if verbose:
                    print(f"{col.values[0]}|{row.values[0]}|{corr_val}")
                
                if col.values[0] not in high_corr.keys():
                    high_corr[col.values[0]] = [[row.values[0],corr_val]]
                else:
                    high_corr[col.values[0]].append([row.values[0],corr_val])
                    
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    if drop_cols:
        drops = set(drop_cols)
        x = x.drop(columns=drops)

    return x, high_corr

def high_corr_label(x, threshold = 0.6, verbose = False):

    # Calculate the correlation matrix
    corr_matrix = abs(x.corr())
    label_corr_feat = []
    label_corr_val = []
    
    for feat in corr_matrix['label'].index:
        
        if feat == 'label':
            continue
        
        corr_val = corr_matrix['label'][feat]
        if  corr_val > threshold:
            
            # Print the correlated features and the correlation value
            if verbose:
                print(f"{feat}:{corr_val}")
            
            corr_val = round(corr_val,2)
            label_corr_feat.append(feat)
            label_corr_val.append(corr_val)
    
    label_corr_feat_arr = np.array(label_corr_feat, dtype=object)
    label_corr_val_arr = np.array(label_corr_val)
    
    order_ind = np.argsort(label_corr_val_arr)
    
    label_corr_feat_arr = label_corr_feat_arr[order_ind]
    label_corr_val_arr = label_corr_val_arr[order_ind]
    
    label_corr = {}
    for key, val in zip(label_corr_feat_arr,label_corr_val_arr):
        label_corr[key] = val
    
    return label_corr_feat_arr, label_corr_val_arr, label_corr

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
    
    def generate_stats_calc(self):
        
        init_base_stats = np.multiply(np.ones(self.args.stats_len),-1)
        
        if self.args.glcm_only == "True":
            stats = {
                'homogeneity': np.copy(init_base_stats),
                'contrast'   : np.copy(init_base_stats),
                'energy'     : np.copy(init_base_stats),
                'correlation': np.copy(init_base_stats)
            }
            
            stats_calc = {
                'homogeneity': self.get_glcm_metrics,
                'contrast'   : self.get_glcm_metrics,
                'energy'     : self.get_glcm_metrics,
                'correlation': self.get_glcm_metrics
            }
        else:
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
        
        return channel_stats, stats_calc, stats
    
    def get_stats(self, clss_imgs, lbl_clss):
        # img is a single channel
        
        channel_stats, stats_calc, stats = self.generate_stats_calc()
        
        channels = {0: 'r', 1: 'g', 2: 'b'}
        hsv = {0: 'h', 1: 's', 2: 'v'}
        for file in tqdm(clss_imgs):
            
            # get indx of unfilled stat
            try:
                available_indxs = np.where(channel_stats['r']['min']==-1)[0]
            except:
                available_indxs = np.where(channel_stats['r']['homogeneity']==-1)[0]
            
            if not available_indxs.any():
                break
            stats_indx = np.min(available_indxs)
            
            img_path = os.path.join(self.imgs_dir,file)
            img = plt.imread(img_path)
            # RGB
            for idx in range(img.shape[-1]):
                channel = img[..., idx]
                channel_glcm = self.get_glcm(channel)
                np_args = {'a':channel} #'a' for 'array' (numpy docs)
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
        
        plt.figure(figsize = (8,6))
        plt.title(f"Examples of {lbl_clss} images")
        
        def plot_img(num_imgs,i,img,text):
            plt.subplot(num_imgs,8,i, xticks = [], yticks = [])
            plt.imshow(img)
            plt.title(f"{text}", fontsize=8)
        
        plt_img_counter = 1
        total_img_counter = 1
        channels = {0: 'R', 1: 'G', 2: 'B'}
        hsv = {0: 'H', 1: 'S', 2: 'V'}
        for file in clss_rdm_imgs:
            img_path = os.path.join(self.imgs_dir,file)
            img = plt.imread(img_path)
            
            plot_img(num_imgs,plt_img_counter,img,f"RBG:{total_img_counter}")
            plt_img_counter += 1
            
            for idx in range(img.shape[-1]):
                channel = img[..., idx]
                plot_img(num_imgs, plt_img_counter, channel, f"{channels[idx]}:{total_img_counter}")
                plt_img_counter += 1
                
            gray_img = self.rgb2gray(img)
            plot_img(num_imgs, plt_img_counter, gray_img, f"gray:{total_img_counter}")
            plt_img_counter += 1
            
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            for idx in range(img_hsv.shape[-1]):
                channel = img_hsv[..., idx]
                plot_img(num_imgs, plt_img_counter, channel, f"{hsv[idx]}:{total_img_counter}")
                plt_img_counter += 1

            total_img_counter += 1
            
        plt.rcParams.update({'font.size': 8})
        
        #plt.show()
        plt.savefig(f"./plots/rgbimgs_{lbl_clss}.png",dpi=200)
        plt.clf()
        
    def plt_gray(self, clss_rdm_imgs, lbl_clss, num_imgs):
        
        def plot_img(num_imgs,i,img,text):
            plt.subplot(num_imgs,1,i)
            plt.imshow(img)
            plt.title(f"{text} {i}")
        
        for counter, file in enumerate(clss_rdm_imgs):
            img_path = os.path.join(self.imgs_dir,file)
            img = plt.imread(img_path)
            
            plot_img(num_imgs,counter+1,img,f"{lbl_clss}")

        plt.show()
        
    # *** MEAN IMAGE ***
    def calc_mean_img(self, path, list_of_filename, size = (256, 256, 3)):
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
        Returns file names of images that belong only to one class
        """
        imgs_indx = np.where(labels==indx)[0]
        class_imgs = imgs[imgs_indx]
        
        return class_imgs
    
    def class_mean_img(self, indx, imgs, labels):
        
        class_imgs = self.get_class_imgs(indx,imgs,labels)
        class_mean_img = self.calc_mean_img(self.imgs_dir,class_imgs)
        
        return class_mean_img
    
    # *** EIGEN IMAGES **
    def img2np(self, path, list_of_filename, size = (256, 256, 3)):
        
        # init full list of images array
        res_fac = self.args.eigen_resize_factor
        a_s = (int(size[0]/res_fac),int(size[1]/res_fac)) # adjusted_size 128,128
        full_mat = np.zeros((self.args.eigen_img_counter,a_s[0]*a_s[1]))
        
        # iterating through each file
        real_counter = 0
        for idx, fn in enumerate(tqdm(list_of_filename)):
            fp = os.path.join(path,fn)
            # read image
            img = plt.imread(fp)
            
            if img.shape == size:
                if self.args.rgb_imgs == "True":
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
    
    def plot_pca(self, pca, lbl_clss, size = (128, 128, 3)):
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
        #splt.show()
        plt.savefig(f"./plots/pca_{lbl_clss}.png",dpi=200)
        plt.clf()
