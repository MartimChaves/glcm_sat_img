# Oil Palm Plantations Satellite Images Classification using GLCMs
## Short Description
This project came about when I was writing a blogpost on Gray Level Co-variance Matrices - you can find it [here](https://medium.com/@mgrc99/glcms-a-great-tool-for-your-ml-arsenal-7a59f1e45b65). I wanted to demonstrate that ***GLCMs were an interesting method for image analysis***.

The majority of the code here is dedicated to **extracting features from a dataset** and using them to **predict their class** using a **classifier**. That dataset is the one provided by the Women in Data Science (WiDS) Datathon 2019, which can be found [here](https://www.kaggle.com/c/widsdatathon2019/data).

Then, in the /fastapi_app directory, you can find the FastAPI API developed, in the inference.py file.

The *overall pipeline* is the following:
1. Exploratory Data Analysis using *features_palm_oil.py*
2. Testing different classifiers using *test_classifiers.py*
3. Grid search the best hyper-parameters for the best classifier and get test set results using *train.py*
4. Use the model to build an API
5. Containerize the model using Docker
6. Deploy the API to the cloud

You can find some of the **results**, **graphs**, and other info in the **plots folder**. The results weren't *mind-blowing*, but still, I thought they were *reasonable*! And ultimately, it was quite a fun project. For the future I was thinking of deploying this model in a web app, which is something that I'd like to learn. *If you'd like to contribute, feel free to shoot me a message, I'd love to collaborate with different people on this project :)*

Side note: in the my_glcm folder you can find a short algorithm to calculate the GLCM of two example images. This is not entirely relevant to the main purpose of this repo, but I thought it was interesting enough to keep it here.

## Install and Run Project

Let's go over the steps to install and run this project.

### 1. Create a virtual environment (optional)

Personally, I recommend that you start by creating a virtual environment for the requirements. Any Python 3.6+ version should work! (below, X stands for your preferred python version - I used 3.8)

- Install virtual environment

> python3.X -m pip install virtualenv

- Create virtual environment

> python3.X -m venv \[environment_name]

- Activate environment

> source \[environment_name]/bin/activate

- Install required libraries

> python3.X -m pip install -r requirements.txt

There may be some unecessary libarires... **After this you should have a working environment that contains all of the required libraries!**

### 2. Create the data directory

Afterward, you need to add the data directory. A directory could look like this:

> ./data/\[dataset_name]/train_images/...

> ./data/\[dataset_name]/train.csv

In the first directory, the *train_images* one, you should have **all of your images** in a format that can be read by matplotlib.pyplot.imread(). The names of the images are not relevant.

In the second directory, the *train.csv* file should contain two columns: one whose column header is 'image', and another whose column header is 'label'. The 'image' column should contain the names of the images in the *train_images* directory, and the 'label' should contain their class. See below:

|    image    | label |
| :--------:  | :---: |
|img_00001.jpg|	  0   |
|img_00002.jpg|	  0   |
|img_00007.jpg|	  1   |
|img_00008.jpg|	  0   |
|     ...     | ...   |

Of the WiDS Datathon 2019 data, I only used the **train data for simplicity**.

Quick side note: I noticed that the *image names* in the *traininglabels.csv* file had a 2017 at the end of them. **Instead of img_00001.jpg, it was img_000012017.jpg**, which took me more time to understand than what I would like to admit... So I just read that csv file, removed the '2017' before the '.jpg' and saved it.

Here are complete instructions to use the WiDS Datathon 2019 data without issues.
1. Download the data from the [Kaggle competition](https://www.kaggle.com/c/widsdatathon2019)
2. Extract the files from the folder
3. From the `widsdatathon2019` folder, copy the `train_images` folder (might be inside another folder called `train_images` as well)  and the `traininglabels.csv` file into the `widsfixed` folder
4. Inside the `widsfixed` directory, run the `create_labels_csv.py` script

**Now you should have some solid data, ready to be used.**

### 3. Carry out EDA (optional, but highly recommended)

For the first step, EDA, you can run the *features_palmoil.py* file. To do so, simply use the following command:

> python3.X features_palmoil.py

If you'd like to change any of the default arguments, for example, if you don't want to plot images of each class, and you want to increase the number of images for the mean image calculation, you can run:

> python3.X features_palmoil.py --eda_rdm_img "False" --mean_img_counter 30

Each argument has a description - it can be accessed by running:

> python3.X features_palmoil.py --h

Currently, this should only work for RGB images. According to the arguments that you parse in, here's what this file can do:

- Plot random images from each class
- Plot the mean image of each class
- Plot the subtraction of the mean of the class 0 from the mean of other classes
- Plot the main Eigen images of each class
- Plot a histogram of each feature
- Plot a KDE plot of relevant features¹
- Plot a pairplot of the features with the highest correlation with the label
- Save a .csv file with the features that correlate the highest between themselves
- Save a .csv file with the correlation values between features and the label

(1) By relevant features, it's understood features that don't have a high correlation with other features

**After doing this, in your plots folder you should find some fresh graphs!**

### 4. Test different classifiers (optional, but highly recommended)

After doing EDA, you should have **some ideas on which features are best**. To change the features that you want to use you have to change the code manually (something to improve later on). 

Go over to the *palm_oil_ds.py* file and in the constructor of the dataset class *add or remove* the features that you want following this naming scheme:

- **"c_measure"**, where 'c' is the channel (r, g, b, g, h, s, v) you want, and 'measure' is the measure you want ('homogeneity', 'energy', 'contrast', 'correlation')²

*If you're using the oil palm plantations dataset, you dont have to worry about this.* To test different classifiers, simply run:

> python3.X test_classifiers.py

The default dataset split is 70% trainset, 20% valset, and 10% testset. If you want, you can change that using the --dataset_split argument. The classifiers that are tested (with pretty much the default hyper-parameters) are the following:

- Random Forest
- Logistic Regression
- Gradient Boosting
- SVM
- KNN

If you'd like to change the hyper-parameters of any of these, you can easily do so in the 'classifiers_dict'. After running this file, it will plot a graph containing the mean and standard deviation of the folds results for three different metrics:

- Balanced accuracy
- AUC
- F1-score

(2) - I've noticed there's a mistake where sometimes I confuse 'energy' with 'entropy'. This is to be fixed in the future.

*After that graph is plotted, you can decide which classifier you want to use.*

### 5. Train a classifier

To train a classifier, run the *train.py* file:

> python3.X train.py

Currently, the **only classifier implemented is the KNN**. This python script will start by running a grid search for the hyper-parameters of the KNN, and afterward the best hyper-parameter configuration will be saved, according to the best F1-score, and used on the testset. A plot of the ROC and a confusion matrix with the testset results will be saved.

There's a lot of future work for this file. For example, it would be great to implement an option to chose different classifiers and which hyper-parameters to use for the grid search.

### 6. Test FastAPI API

To test the FastAPI API, change the directory to /fastapi_app and run:

> uvicorn inference:app --reload

### 7. Test API with Docker

To test API with Docker, you'll have to build the image and then run it.
Command to build image:

> docker build -t oil-palm-inference .

Command to run it:

> docker run -p 8000:8000 oil-palm-inference

### 8. Deploy API to the Cloud

To deploy the API to AWS using Elastic Beanstalk (EB), follow these steps:

1. Initialize an EB environment

> eb init -p docker -r eu-west-1 oil-palm-serving

2. Create an EB environment

> eb create oil-palm-serving-env

The URL that shows up in the terminal after the line "INFO Application available at ..." is the URL where the application will be available.

3. Terminate the EB environment when you're done

> eb terminate oil-palm-serving-env

## Using the Project for Different Datasets

I would say that this project is very easily adaptable to other datasets - you simply have to parse in the right data root when running each file. 

However, the *mean_img* and *eigen_img* method of the *DS_aux* class will not work without changing the code directly - there are some image sizes that have to be manually changed (to be improved in the future), so you might want to parse those args as "False" (--eda_mean_img and --eda_eigen_img). There may be some other locations where you might have to change the code, but it should be easy enough. The remaining code should work (if not, shoot me a message!).

### Contact and Questions

If you have any question, feel free to ask at: mgrc99@gmail.com

If this helped you in any way, consider liking my medium posts:

1. [On GLCMs](https://medium.com/@mgrc99/glcms-a-great-tool-for-your-ml-arsenal-7a59f1e45b65)
2. [On FastAPI & Model Deployment](https://medium.com/@mgrc99/how-to-deploy-a-fast-and-scalable-image-classification-web-app-538b1f43fb22#1826)

Thank you!
