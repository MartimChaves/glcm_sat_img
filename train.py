import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

from features_palmoil import DS_aux

import argparse
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from timeit import default_timer as timer

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
            
            self.train[...,i] = self.train[...,i] - feat_mean
            self.train[...,i] = self.train[...,i] / feat_std
            
            self.val[...,i] = self.val[...,i] - feat_mean
            self.val[...,i] = self.val[...,i] / feat_std
            
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

    def calc_clss_weights(self):
        weight_dict = {}
        total_num_samples = len(self.train_labels) 
        for clss_lbl, clss_val in self.label_code.items():
            clss_num_samples = len(np.where(self.train_labels == clss_val)[0])
            weight_dict[clss_val] = 1 - (clss_num_samples/total_num_samples)
        return weight_dict

def main(args):
    dataset = PalmOilDataset(args)
    print("Generating dataset...")
    dataset.generate_features('fold_1')
    print("Dataset generated...")
    
    weight_dict = dataset.calc_clss_weights()
    
    classifiers_dict = {
        'random_forest': {
            'model': RandomForestClassifier,
            'model_args': {'max_depth':2,'class_weight':weight_dict,'random_state':args.seed},
            'val_preds': 0,
            'bal_acc': [],
            'auc': []
        },
        'log_reg': {
            'model': LogisticRegression,
            'model_args': {'class_weight':weight_dict,'random_state':args.seed},
            'val_preds': 0,
            'bal_acc': [],
            'auc': []
        },
        'gradient_boost': {
            'model': GradientBoostingClassifier,
            'model_args': {'random_state':args.seed},
            'val_preds': 0,
            'bal_acc': [],
            'auc': []
        },
        'svm': {
            'model': SVC,
            'model_args': {'probability':True,'class_weight':weight_dict,'random_state':args.seed},
            'val_preds': 0,
            'bal_acc': [],
            'auc': []
        },
        'knn': {
            'model': KNeighborsClassifier,
            'model_args': {'n_neighbors':3},
            'val_preds': 0,
            'bal_acc': [],
            'auc': []
        },
        'dummy': {
            'model': DummyClassifier,
            'model_args': {'strategy':"most_frequent",'random_state':args.seed},
            'val_preds': 0,
            'bal_acc': [],
            'auc': []
        }
    }
    
    print("Training classifiers...")
    for clssfier_name, classifier in classifiers_dict.items():
        print(f"******************{clssfier_name}********************")
        clf = classifier['model'](**classifier['model_args'])
        
        start = timer()
        clf.fit(dataset.train, dataset.train_labels)
        end = timer()
        time_elapsed = end-start
        print(f"Time elapsed for {clssfier_name}:{round(time_elapsed,2)}s")
        
        classifier['val_preds'] = clf.predict(dataset.val)
        bal_acc_val = round(balanced_accuracy_score(dataset.val_labels, classifier['val_preds'])*100,2)
        classifier['bal_acc'].append(bal_acc_val)
        print(f"Balanced accuracy {clssfier_name}:{bal_acc_val}%")
        
        raw_probs = clf.predict_proba(dataset.val)
        auc = roc_auc_score(dataset.val_labels, raw_probs[:, 1])
        classifier['auc'].append(auc)
        print(f"AUC {clssfier_name}:{round(auc,5)}")
        
        # f1-score
        
    
    # balanced accuracy
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    