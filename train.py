import numpy as np
import pandas as pd
import argparse
from palm_oil_ds import PalmOilDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, roc_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt
import seaborn as sns

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
    dataset.generate_features('fold_1', gen_full_data=True)
    
    print("Determining best hyper-parameters for classifier...")
    clf = KNeighborsClassifier()
    parameters = {'weights':('uniform', 'distance'), 'n_neighbors':[3, 5], 'p':[1,2]}
    
    grid_kwargs = {'scoring':'f1',
                   'n_jobs':-1,
                   'cv':dataset.num_folds,
                   'refit':True}
    
    grid_clf = GridSearchCV(clf, parameters, **grid_kwargs)
    grid_clf.fit(dataset.full_data, dataset.full_data_labels)
    
    print(f"Best parameters for classifier:{grid_clf.best_params_}")
    
    dataset.gen_test_set()
    
    pred_test_labels = grid_clf.predict(dataset.test)
    raw_probs = grid_clf.predict_proba(dataset.test)[:, 1]
    
    auc = round(roc_auc_score(dataset.test_labels, raw_probs)*100,2)
    fpr, tpr, _ = roc_curve(dataset.test_labels, raw_probs)
    
    bal_acc_test = round(balanced_accuracy_score(dataset.test_labels, pred_test_labels)*100,2)
    f1 = round(f1_score(dataset.test_labels, pred_test_labels, average='binary')*100,2)
    
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curve - Sat images")
    plt.legend(loc="lower right")
    plt.savefig("./plots/roc.png")
    plt.clf()
    
    confmatrix = confusion_matrix(dataset.test_labels,pred_test_labels)
    cmat_size = len(confmatrix)
    df_cm = pd.DataFrame(confmatrix, range(cmat_size), range(cmat_size))
    fig = plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="gray",linewidths=0.1, linecolor='gray') # font size
    
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.title('Test Set Confusion Matrix')
    fig.savefig(f'./plots/f1_{f1}_bal_acc_{bal_acc_test}_confmat_test.png', dpi = 150)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)