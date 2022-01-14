import numpy as np
import os
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
    
    def norm_features(self):
        
        # Per-fold normalization
        
        """
        self.norm_vals_dict = {
            'fold_1': {
                'feat_1':[mean, std],
                'feat_2':[mean, std]
            },
            'fold_2': {
                'feat_1':[mean,std],
                ...
            },
            ...
        }
        
        # test set is normalized with the fold that has the best metrics
        
        """
        pass
    
    def calculate_features(self, img):
        
        stats_calc = {
                'r_energy'     : np.min,
                'r_correlation': np.max,
                'r_contrast'   : np.mean,
                'r_homogeneity': np.std,
                's_correlation': self.get_glcm_metrics,
                'g_energy'     : self.get_glcm_metrics,
                'h_correlation': self.get_glcm_metrics,
                's_contrast'   : self.get_glcm_metrics
            }
        
        features = np.zeros((8))
        
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
    
    def generate_features(self):
        # similar to img_utils func but to all of the imgs
        
        fold_features = {}
        
        for fold, fold_info in self.folds.items():

            fold_features[fold] = {}
            
            init_train_feats = np.zeros((len(fold_info['train']),8)) # 8 features were chosen
            init_val_feats = np.zeros((len(fold_info['val']),8))

            print(f"Calculating train set features for {fold}")
            init_train_feats = self.calc_set_feats(fold_info['train'], init_train_feats)
            
            print(f"Calculating validation set features for {fold}")
            init_val_feats = self.calc_set_feats(fold_info['val'], init_val_feats)
            
            fold_features[fold]['train'] = init_train_feats
            fold_features[fold]['val'] = init_val_feats
            fold_features[fold]['train_labels'] = fold_info['train_labels']
            fold_features[fold]['val_labels'] = fold_info['val_labels']
        
        
        # similar to dataset split, but instead of img file, img features
        # feat_vals = {
            # 'fold_1' :
        # }
        
        
        pass
    
    
    
    
def main(args):
    dataset = PalmOilDataset(args)
    dataset.generate_features()
    

if __name__ == "__main__":
    args = parse_args()
    main(args)