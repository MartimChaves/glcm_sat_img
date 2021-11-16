# imports
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Config')
    
    parser.add_argument('--root', type=str, default='./data/apl_trees/plant-pathology-2021-fgvc8/', help='Directory where files are.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random numbers.')
    
    # Data config
    parser.add_argument('--dataset_split', type=str, default='70,20,10', help='Percentage of dataset allocated to each set (train,val,test).')
    
    # EDA
    parser.add_argument('--eda', type=str, default='True', help='If true, carry out exploratory data analysis (EDA).')
    parser.add_argument('--eda_rdm_img', type=str, default='True', help='If true, show random images (EDA).')
    parser.add_argument('--eda_rdm_img_num', type=int, default=6, help='Number of random images to show.')
    
    args = parser.parse_args()
    return args

class Img_Features():
    
    def __init__(self,args):
        
        self.args = args
        self.imgs_dir = os.path.join(self.args.root,"train_images")
        #self.extra_imgs_dir = os.path.join(self.args.root,"test_images")
        
        self.data_split()
        
        return

    def data_split(self):
        
        img_files = os.listdir(self.imgs_dir)
        np.random.shuffle(img_files)
        
        # get split fractions
        trainset_fraction = int(self.args.dataset_split.split(",")[0])/100.
        valset_fraction = int(self.args.dataset_split.split(",")[1])/100.
        testset_fraction = int(self.args.dataset_split.split(",")[2])/100.
        
        trainset_len = int(len(img_files)*trainset_fraction)
        valset_len = int(len(img_files)*valset_fraction)
        testset_len = len(img_files) - trainset_len - valset_len
        
        self.testset_files = img_files[-testset_len::]
        img_files = img_files[0:-testset_len]
        
        # cross-validation
        k_num = int((trainset_fraction+valset_fraction)/valset_fraction)
        
        self.folds = []
        for i in range(k_num):
            val_files = img_files[i*valset_len:(i+1)*valset_len]
            train_files = [file for file in img_files if file not in val_files]

            self.folds.append([train_files,val_files])
            
        return
    
    ### exploratory data analysis
    def show_rdm_imgs(self):
        return

    def eda(self):
        data_root = self.args.root

        # random images
        if self.args.eda_rdm_img == "True":
            train_imgs_files = os.listdir(self.train_imgs_dir)
            test_imgs_files = os.listdir(self.test_imgs_dir)

            num_imgs = self.args.eda_rdm_img_num
            
            train_imgs = np.random.choice(train_imgs_files,num_imgs//2,replace=False)
            test_imgs = np.random.choice(test_imgs_files,num_imgs//2,replace=False)
            
            plt.figure(figsize = (8,6))
            
            for i in range(1,num_imgs+1):
                if i <= num_imgs//2:
                    img_file = os.path.join(self.args.root,"train_images",train_imgs[i-1])
                    img = plt.imread(img_file)
                    plt.subplot(2,num_imgs//2,i)
                    plt.imshow(img)
                    plt.title("Train image " + str(i))
                else:
                    j = i-num_imgs//2
                    img_file = os.path.join(self.args.root,"test_images",test_imgs[j-1])
                    img = plt.imread(img_file)
                    plt.subplot(2,num_imgs//2,i)
                    plt.imshow(img)
                    plt.title("Test image " + str(j))
            
            plt.show()
            
        # average image for each class
        # ravel image into vector
        # reshape to original size

        # difference between average of images

        # variability 

        # eigenfaces

        # RGB/HSV/Grey
        # Max value, min value, mean, std for each class
        # correlation for each class and in general


        return

### train val test split

### feature development

### feature selection


def main(args):
    
    np.random.seed(args.seed)
    
    Features = Img_Features(args)
    
    if args.eda == "True":
        Features.eda()
    
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)