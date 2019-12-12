# AI Programming with Python Project

Second project in Udacity's Artificial Intelligence Programming with Python Nanodegree program. Used a pre-trained Convolutional Neural Network (CNN) developed with PyTorch. The pre-trained CNN model extracts image features. Built and trained classifier of the model to recognize different species of flowers. A data-set of 102 flower categories was used. The project consisted of two parts, the first part consists of a jupyter notebook with the initial version of the classifier and the second part is a command line application. The command line application consists of two files train.py and predict.py command line interfaces. The first file, train.py, trains a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. The following model architectures are supported vgg16, alexnet, densenet161, and mobilenet_v2.

## Getting Started 

1. Clone the repository
```console
git clone https://github.com/atkatchev/ 
cd DLND_Project1_workspace 
```
2. Open the Project file using jupyter notebook. Otherwise, open .html page
```console
jupyter notebook Your_first_neural_network.ipynb
```
3. Use SHIFT ENTER keys to run each cell
4. Running the command line application please use below;
Help: python train.py -h
- Basic usage: python train.py data_directory
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg16 OR alexnet OR densenet161 OR mobilenet_v2"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 4096 --epochs 32
- Use GPU for training: python train.py data_dir --gpu

Help: python predict.py -h
- Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
- Basic usage: python predict.py /path/to/image checkpoint
- Return top K most likely classes: python predict.py input checkpoint --top_k 3
- Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python predict.py input checkpoint --gpu


## Installation
-    python 3.5 
-    Pandas
-    Numpy
-    Matplotlib
-    Scikit-Learn
-    PyTorch
-    torchvision
-    PIL
-    json
-    collections

## Structure 
- Image Classifier Project.ipynb - first part of project containing functionality of predict.py and train.py 
- Image Classifier Project.html - html equivalent of the above ipynb (jupyter notebook)
- cat_to_name.json - dictionary mapping from category label to category name, integer encoded categories to the actual names of the flowers.
- predict.py - trains a network on a dataset and saves the model as a checkpoint  
- train.py - uses a trained network to do inference (predict the class for an input image)

## Dataset

Download the dataset used for this project by typing the commands below:
```
wget -c https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
unzip -qq flower_data.zip
```

## Checkpoints
Have been removed due to github file size limit of 100.00 MB
