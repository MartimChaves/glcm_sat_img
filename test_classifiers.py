import numpy as np
from matplotlib import pyplot as plt

from palm_oil_ds import PalmOilDataset

import argparse

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

from timeit import default_timer as timer

def parse_args():
    parser = argparse.ArgumentParser(description='Config')
    
    parser.add_argument('--root', type=str, default='./data/widsfixed/', help='Directory where files are.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random numbers.')
    
    # Data config
    parser.add_argument('--dataset_split', type=str, default='70,20,10', help='Percentage of dataset allocated to each set (train,val,test).')
    
    args = parser.parse_args()
    return args

def main(args):
    dataset = PalmOilDataset(args)
    metrics_used = ['bal_acc','auc','f1']
    
    weight_dict = 0 # to be determined
    classifiers_dict = {
        'random_forest': {
            'model': RandomForestClassifier,
            'model_args': {'max_depth':2,'class_weight':weight_dict,'random_state':args.seed}
        },
        'log_reg': {
            'model': LogisticRegression,
            'model_args': {'class_weight':weight_dict,'random_state':args.seed}
        },
        'gradient_boost': {
            'model': GradientBoostingClassifier,
            'model_args': {'random_state':args.seed}
        },
        'svm': {
            'model': SVC,
            'model_args': {'probability':True,'class_weight':weight_dict,'random_state':args.seed}
        },
        'knn': {
            'model': KNeighborsClassifier,
            'model_args': {'n_neighbors':3}
        },
        'dummy': {
            'model': DummyClassifier,
            'model_args': {'strategy':"most_frequent",'random_state':args.seed}
        }
    }
    
    for _, classifier in classifiers_dict.items():
        for metric in metrics_used:
            classifier[metric] = []
    
    print("Training classifiers...")
    for i in range(dataset.num_folds):
        print("Generating dataset...")
        dataset.generate_features(f'fold_{i+1}')
        print("Dataset generated...")
        weight_dict = dataset.calc_clss_weights()
        
        for clssfier_name, classifier in classifiers_dict.items():
            print(f"****************** Fold {i+1} - {clssfier_name} ********************")
            
            if 'class_weight' in classifier['model_args']:
                classifier['model_args']['class_weight'] = weight_dict
            
            clf = classifier['model'](**classifier['model_args'])
            
            start = timer()
            clf.fit(dataset.train, dataset.train_labels)
            end = timer()
            time_elapsed = end-start
            print(f"Time elapsed for {clssfier_name}:{round(time_elapsed,2)}s")
            
            # validation set predictions
            pred_val_labels = clf.predict(dataset.val)
            raw_probs = clf.predict_proba(dataset.val)[:, 1]
            
            bal_acc_val = round(balanced_accuracy_score(dataset.val_labels, pred_val_labels)*100,2)
            classifier['bal_acc'].append(bal_acc_val)
            print(f"Balanced accuracy {clssfier_name}:{bal_acc_val}%")
            
            auc = round(roc_auc_score(dataset.val_labels, raw_probs)*100,2)
            classifier['auc'].append(auc)
            print(f"AUC {clssfier_name}:{auc}")
            
            f1 = round(f1_score(dataset.val_labels, pred_val_labels, average='binary')*100,2)
            classifier['f1'].append(f1)
            print(f"F1-score {clssfier_name}:{f1}")
    
    
    print(f"######## Metrics ########")
    score_dicts = {}
    for metric in metrics_used:
        score_dicts[metric] = [[],[]] # score and error
    for clssfier_name, classifier in classifiers_dict.items():
        print(f"****************** {clssfier_name} ********************")
        for metric in metrics_used:
            mean = np.mean(classifier[metric])
            std = round(np.std(classifier[metric]),4)
            
            score_dicts[metric][0].append(mean)
            score_dicts[metric][1].append(std)
            
            print(f"* {metric}: mean = {mean}; std = {std}")
    
    y = np.array(score_dicts['bal_acc'][0][0:-1])
    e = np.array(score_dicts['bal_acc'][1][0:-1])

    y_2 = np.array(score_dicts['auc'][0][0:-1])
    e_2 = np.array(score_dicts['auc'][1][0:-1])

    y_3 = np.array(score_dicts['f1'][0][0:-1])
    e_3 = np.array(score_dicts['f1'][1][0:-1])

    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4       # the width of the bars

    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot(111)

    style_kwargs = {'linestyle':'None', 'capsize':4, 'ecolor':'#D81B60', 'fmt':'o', 'markersize':6, 'elinewidth':0.8}

    errplot_1 = ax.errorbar(ind, y, e, color='#1E88E5',**style_kwargs)
    errplot_2 = ax.errorbar(ind+(width/2), y_2, e_2, color='#FFC107',**style_kwargs)
    errplot_3 = ax.errorbar(ind+width, y_3, e_3, color='#004D40',**style_kwargs)

    ax.set_ylabel('Scores', fontsize=15)
    ax.set_title('Metric Scores for Different Classifiers', fontsize=20, pad=17)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels( ('Random\nForests', 'Logistic\nRegression', 'Gradient\nBoosting', 'SVM', 'KNN') )

    ax.legend( (errplot_1[0], errplot_2[0], errplot_3[0]), ('Balanced Accuracy', 'AUC', 'F1-Score') )
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

    plt.savefig("./plots/classifier_metrics.png",dpi=200)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    