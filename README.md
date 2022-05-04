# Face-Mask Detection project

## About Project
This project uses a Deep Neural Network, more specifically DenseNet201, to differentiate between images of people with and without masks. The model get an accuracy of 99.6% on the training set and 99.5% on the test set. Then the stored model used to classify as mask or no mask.
## Dataset

The dataset used can be downloaded from [here](https://www.kaggle.com/code/shahf11/face-mask-detection-with-densenet201-99-6-acc/data), the dataset consists of almost 12K images which are almost 328.92MB in size.

** the dataset in this repository is a sample from the original data 

This dataset is already divided into three chunks (train, test, validation):
* 10000 images as a train set:
* 800 images as a validation set:
* 992 images as a test set


## How to Use

To use this project on your system, follow the following steps:

1. Clone this repository onto your system by typing the following command on your Command Prompt

2. Download all libraries:
    
   Using pip
   ```bash
   pip install -r requirements.txt
   ```
   Using Anaconda
   ```bash
   conda create --name env_name --file requirements.txt
   ```
3. Run facemask.py by typing the following command on your command line:
    ```bash
    python detect.py [--image "image/path"][--mode "train/predict"]
    ```
   notes:
      * to run the script you need to specify the mode, we have two modes:
        * **predict**: which used in case you need to the prediction from a specific image using the saved model(pre-trained) 
        * **train**: used in case you need train the model on new data
      * image path needed if the mode is "predict" 
      * the default mode is "predict"