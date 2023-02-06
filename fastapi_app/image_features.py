import numpy as np
import pandas as pd
import cv2

import sys
sys.path.append("..")
from img_utils import Image_Funcs


class ImageFeatures(Image_Funcs):
    
    def __init__(self):
        
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
    
    def norm_features(self, features, stats_csv="norm_stats.csv"):
        norm_stats_df = pd.read_csv(stats_csv)
        
        mean_arr = np.array(norm_stats_df["mean"])
        std_arr = np.array(norm_stats_df["std"])
        
        for idx, feat in enumerate(features):
            temp_feat = (feat-mean_arr[idx])/std_arr[idx]
            features[idx] = temp_feat

        return features
