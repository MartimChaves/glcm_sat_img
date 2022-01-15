import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

from features_palmoil import DS_aux

import argparse
from tqdm import tqdm
"""
r_energy	0.21
r_correlation	0.25
r_contrast	0.28
r_homogeneity	0.34
s_correlation	0.38
g_energy	0.4
h_correlation	0.41
s_contrast	0.66
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Config')
    
    parser.add_argument('--root', type=str, default='./data/widsfixed/', help='Directory where files are.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random numbers.')
    
    # Data config
    parser.add_argument('--dataset_split', type=str, default='70,20,10', help='Percentage of dataset allocated to each set (train,val,test).')
    
    args = parser.parse_args()
    return args


class PalmOilDataset(DS_aux):
    
    def __init__(self, args,
                 label_code={'No_OilPalm':0,
                             'Has_OilPalm':1}):
        super().__init__(args,label_code)
        
        self.stats_calc = {
                'r_energy'     : self.get_glcm_metrics,
                'r_correlation': self.get_glcm_metrics,
                'r_contrast'   : self.get_glcm_metrics,
                'r_homogeneity': self.get_glcm_metrics,
                'g_energy'     : self.get_glcm_metrics,
                'h_correlation': self.get_glcm_metrics,
                's_correlation': self.get_glcm_metrics,
                's_contrast'   : self.get_glcm_metrics
            }
    
    def norm_features(self):
        
        self.feats_mean = []
        self.feats_std = []
        
        for i in range(self.train[0].shape[0]): #number of features
            feat_mean = np.mean(self.train[...,i])
            feat_std = np.std(self.train[...,i])
            
            self.train[0::][0] = self.train[...,i] - feat_mean
            self.train[0::][0] = self.train[...,i] / feat_std
            
            self.val[0::][0] = self.train[...,i] - feat_mean
            self.val[0::][0] = self.train[...,i] / feat_std
            
            self.feats_mean.append(feat_mean) # save in case it's needed for testset
            self.feats_std.append(feat_std)
        
    
    def calculate_features(self, img):
        features = np.zeros((8))
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        glcm_dict = {
            'r':self.get_glcm(img[...,0]),
            'g':self.get_glcm(img[...,1]),
            'h':self.get_glcm(img_hsv[...,0]),
            's':self.get_glcm(img_hsv[...,1])
        }
        
        for idx, (stat, calc_func) in enumerate(self.stats_calc.items()):
            channel = stat[0]
            channel_glcm = glcm_dict[channel]
            feat_val = calc_func(stat[2::],glcm=channel_glcm)
            features[idx] = feat_val
        
        return features
    
    def calc_set_feats(self, img_files, init_set_feats):
        """
        Read images, calculate features, store it in array, return array
        """
        for idx, img_file in enumerate(tqdm(img_files)):
            img_path = os.path.join(self.imgs_dir,img_file)
            img = plt.imread(img_path)
            img_feats = self.calculate_features(img)
            init_set_feats[idx] = img_feats
        return init_set_feats
    
    def generate_features(self, fold):
        fold_info = self.folds[fold]
        
        init_train_feats = np.zeros((len(fold_info['train']),8)) # 8 features were chosen
        init_val_feats = np.zeros((len(fold_info['val']),8))

        print(f"Calculating train set features for {fold}")
        init_train_feats = self.calc_set_feats(fold_info['train'], init_train_feats)
        
        print(f"Calculating validation set features for {fold}")
        init_val_feats = self.calc_set_feats(fold_info['val'], init_val_feats)
        
        self.train = init_train_feats
        self.val = init_val_feats
        self.train_labels = fold_info['train_labels']
        self.val_labels = fold_info['val_labels']
        
        self.norm_features()
    
def main(args):
    dataset = PalmOilDataset(args)
    dataset.generate_features('fold_1')

if __name__ == "__main__":
    args = parse_args()
    main(args)