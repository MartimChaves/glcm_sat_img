import numpy as np
import pandas as pd
import cv2
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

from features_palmoil import DS_aux

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
        
        # Number of features available
        self.n_feats = len(self.stats_calc.keys())
    
    def norm_features(self, train_set, val_set=[],
                      logstats=False, testset=False):

        if logstats:
            stats_df = pd.DataFrame(-1, index=range(self.n_feats),
                                    columns=["mean", "std"])

        if testset:
            read_stats_df = pd.read_csv("./fastapi_app/norm_stats.csv")

        for i in range(self.train[0].shape[0]): #number of features
            if not testset:
                feat_mean = np.mean(train_set[...,i])
                feat_std = np.std(train_set[...,i])

                if logstats:
                    stats_df.at[i, "mean"] = feat_mean
                    stats_df.at[i, "std"] = feat_std
            else:
                feat_mean = read_stats_df["mean"][i]
                feat_std = read_stats_df["std"][i]
            
            train_set[...,i] = train_set[...,i] - feat_mean
            train_set[...,i] = train_set[...,i] / feat_std
            
            if not len(val_set) == 0:
                val_set[...,i] = val_set[...,i] - feat_mean
                val_set[...,i] = val_set[...,i] / feat_std
        
        # Save stats
        if logstats:
            stats_df.to_csv("./fastapi_app/norm_stats.csv")

        if not len(val_set) == 0:
            return train_set, val_set
        else:
            return train_set
    
    def calculate_features(self, img):
        features = np.zeros((self.n_feats))
        
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
    
    def generate_features(self, fold, gen_full_data = False):
        fold_info = self.folds[fold]
        
        init_train_feats = np.zeros((len(fold_info['train']),self.n_feats))
        init_val_feats = np.zeros((len(fold_info['val']),self.n_feats))

        print(f"Calculating train set features for {fold}")
        init_train_feats = self.calc_set_feats(fold_info['train'], init_train_feats)
        
        print(f"Calculating validation set features for {fold}")
        init_val_feats = self.calc_set_feats(fold_info['val'], init_val_feats)
        
        self.train = init_train_feats
        self.val = init_val_feats
        self.train_labels = fold_info['train_labels']
        self.val_labels = fold_info['val_labels']
        
        if gen_full_data:
            self.full_data = np.concatenate((self.train,self.val),axis=0)
            self.full_data = self.norm_features(self.full_data, logstats=True)
            self.full_data_labels = np.concatenate((self.train_labels, self.val_labels),axis=0)
        
        self.train, self.val = self.norm_features(self.train, self.val)

    def gen_test_set(self):
        
        init_test_feats = np.zeros((len(self.imgs_test), self.n_feats)) #labels: self.test_labels
        init_test_feats = self.calc_set_feats(self.imgs_test, init_test_feats)
        self.test = init_test_feats
        self.test = self.norm_features(self.test, testset=True)
        
    def calc_clss_weights(self):
        weight_dict = {}
        total_num_samples = len(self.train_labels) 
        for clss_lbl, clss_val in self.label_code.items():
            clss_num_samples = len(np.where(self.train_labels == clss_val)[0])
            weight_dict[clss_val] = 1 - (clss_num_samples/total_num_samples)
        return weight_dict