# Oil Palm Plantations Satellite Images Classification using GLCMs
### Short Description
This project came about when I was writing a blogpost on Gray Level Co-variance Matrices (you can find it [here]()!). I wanted to demonstrate that ***GLCMs were an interesting method for image analysis***.

The majority of the code here is dedicated to **extracting features from a dataset** and using them to **predict their class** using a **classifier**. That dataset is the one provided by the Women in Data Science (WiDS) Datathon 2019, which can be found [here](https://www.kaggle.com/c/widsdatathon2019/data).

The *overall pipeline* is the following:
1. Exploratory Data Analysis using *features_palm_oil.py*
2. Testing different classifiers using *test_classifiers.py*
3. Grid search the best hyper-parameters for the best classifier and get test set results using *train.py*

You can find some of the **results**, **graphs**, and other info in the **plots folder**. The results weren't *mind-blowing*, but still, I thought they were *reasonable*! And ultimately, it was quite a fun project. For the future I was thinking of deploying this model in a web app, which is something that I'd like to learn. *If you'd like to contribute, feel free to shoot me a message, I'd love to collaborate with different people on this project :)*

Side note: in the my_glcm folder you can find a short algorithm to calculate the GLCM of two example images. This is not entirely relevant to the main purpose of this repo, but I thought it was interesting enough to keep it here.

### Install and Run Project

Let's go over the steps to install and run this project.

Personally, I recommend that you start by creating a virtual environment for the requirements. Any Python 3.6+ version should work! (below, X stands for your preferred python version - I used 3.8)

- Install virtual environment

> python3.X -m pip install virtualenv

- Create virtual environment

> python3.X -m venv \[environment_name]

- Activate environment

> source \[environment_name]/bin/activate

- Install required libraries

> python3.X -m pip install -r reqs_new.txt

**After this you should have a working environment that contains all of the required libraries!**

Afterward, you need to add the data directory. A directory could look like this:

> ./data/\[dataset_name]/\[train_images]/...
> ./data/\[dataset_name]/\[train.csv]

In the first directory, the *train_images* one, you should have **all of your images** in a format that can be read by matplotlib.pyplot.imread(). The names of the images are not relevant.

In the second directory, the *train.csv* file should contain two columns: one whose column header is 'image', and another whose column header is 'label'. The 'image' column should contain the names of the images in the *train_images* directory, and the 'label' should contain their class. See below:

    image      label
img_00001.jpg	 0
img_00002.jpg	 0
img_00007.jpg	 1
img_00008.jpg	 0
...             ...

### Using the Project for Different Datasets

I would say that this project is very easily adaptable to other datasets - you simply have to parse in the right root when running each file. 

However, the *mean_img* and *eigen_img* method of the *DS_aux* class will not work without changing the code directly - there are some image sizes that have to be manually changed (to be improved in the future). The remaining code should work (if not, shoot me a message!).

### Contact and Questions

If you have any question, feel free to ask at: mgrc99@gmail.com

If this helped you in any way, consider liking my [medium post]()! Thank you!
