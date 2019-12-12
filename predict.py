#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEV notes:
uses a trained network to predict the class for an input image
pass in a single image /path/to/image and 
return the flower name and class probability

python predict.py /path/to/image checkpoint
python predict.py input checkpoint --top_k 3
python predict.py input checkpoint --category_names cat_to_name.json
python predict.py input checkpoint --gpu  
"""

import numpy as np
import matplotlib.pyplot as plt # used to show() test image flower_files[3000]                 
import torch 
import torchvision.transforms as transforms #used in transformations 
import json #used to decode json file cat_to_name.json
import torchvision.models as models #imports models, pretrained VGG16 is used  
#import is for training to be robust on truncated images
from PIL import ImageFile #work with JPEG images
ImageFile.Load_TRUNCATED_IMAGES = True
from PIL import Image
import torch.nn.functional as F #used for softmax function
import argparse  #used to get arguments from user 

def get_input_args():
    """
    Retrieves and parses command line arguments provided by the user when
    they run the program from a terminal. This function uses Python's 
    argparse module to create and define these command line arguments 
    If the user fails to provide some or all of the 3 arguments, then the 
    missing arguments.
    1. image --path path to image default 
    2. checkpoint name --chekckpoint name of checkpoint 
    3. amount of classes to return top k --top_k
    4. specify file containg target label names categotry_names 
    5. specify run on gpu --gpu 
    """ 
    #Create Argument Parser object named parser 
    parser = argparse.ArgumentParser()
    #Argument 1: 
    parser.add_argument('--path', type=str, default='flowers/test/69/image_05971.jpg', \
                        help='path/to/image')
    #Argument 2:
    parser.add_argument('--check_point', type=str, default='model_transfer.pt', \
                        help='checkpoint name')
    #Argument 3: 
    parser.add_argument('--top_k', type=int, default='1', \
                        help='amount of classes to return')
    #Argument 4: 
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', \
                        help='file with target labels')
    #Argument 5: 
    parser.add_argument('--gpu', type=str, default='cpu', \
                        help='hardware to run on CPU or GPU')
    #Assign variables in_args tp parse_args()
    in_args = parser.parse_args()
    #access values of Arguments by printing it 
    #print("Arguments: ", in_args) #DEBUG 
    return in_args

def load_model_transfer_checkpoint(checkpoint_path):
    """Load a checkpoint and rebuild the model.

    Parameters
    ----------
    checkpoint_path : str
    Check point path
    """    
    #load the saved checkpoint
    checkpoint = torch.load(checkpoint_path)
    #print(checkpoint) #DEBUG
    #Model Architecure pre trained VGG16
    model_transfer = models.vgg16(pretrained=True)
    #print(model_transfer) #DEBUG
    #Freez feature parameters
    for param in model_transfer.parameters():
        param.requires_grad = False 
    model_transfer.architecture = checkpoint['architecture']
    model_transfer.classifier = checkpoint['classifier']
    model_transfer.load_state_dict = checkpoint['state_dict']
    #optimizer_transfer.state_dict = checkpoint['optimizer_dict']
    model_transfer.class_to_idx = checkpoint['class_to_idx']
    #print(model_transfer) #DEBUG
    return model_transfer

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

    Parameters
    ----------
    path : str
    image path
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #normalize the means and standard deveiations 
    #of the images to what the netword expects
    standard_normalization = transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])
    img = Image.open(image_path)
    
    im_transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      standard_normalization])
    
    processed_image = im_transforms(img)
    
    return processed_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor.
    
    Parameters
    ----------
    path : str
    image path
    """
    #print(title)
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    Parameters
    ----------
    image_path : str
    image path
    model :
    trained model from checkpoint model 
    topk : int
    predict top-K most probable classes
    '''
    #print(image_path, model, topk)
    # TODO: Implement the code to predict the class from an image file
    img_processed = process_image(image_path)
    #add a dimension with a length of one expanding the rank
    img_squeezed = img_processed.unsqueeze_(0)
    #convert to floating point notation
    img_squeezed_fp = img_squeezed.float()
    with torch.no_grad():
        output = model(img_squeezed_fp)
    probability = torch.exp(output)
    #print('output1',probability.topk(topk))
    #probability = F.softmax(output.data,dim=1)
    #return probability.topk(topk)
    return probability.topk(topk)
def check_sanity(image_path, cat_to_name, model_transfer):
    ''' Plot the probabilities for the top 5 classes as a bar graph, along with the input image 
    Parameters
    ----------
    image_path : str
    image path
    cat_to_name : str
    mapping from category label to category name
    model :
    trained model from checkpoint model 
    '''

    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    flower_num = image_path.split('/')[2]#title
    fig_title = cat_to_name[flower_num]#get name of jpg
    axs = imshow(process_image(image_path), ax=plt) #plot fig 1
    axs.suptitle(fig_title)
    #axs.show() 

    probs_classes = predict(image_path, model_transfer)
    #print('probs_classes',probs_classes)
    probs = np.array(probs_classes[0][0])
    #print('probs',probs)
    classes = [cat_to_name[str(index + 1)] for index in np.array(probs_classes[1][0])]
    #print('classes',classes)  
    
    classes_len = float(len(classes))
    fig,ax = plt.subplots(figsize=(7,4))
    width = 0.8
    tick_locs = np.arange(classes_len)
    ax.bar(tick_locs, probs, width, linewidth=4.0, align = 'center')
    ax.set_xticks(ticks = tick_locs)
    ax.set_xticklabels(classes)
    ax.set_xlim(min(tick_locs)-0.6,max(tick_locs)+0.6)
    ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2])
    ax.set_ylim((0,1))
    ax.yaxis.grid(True)
    #plt.subplot(2,2,1)
    plt.show()

def main():
    """
    Main 
    """
    #print('python predict.py -h')
    in_args = get_input_args()
    print(in_args)
    device = torch.device("cuda:0" if in_args.gpu and torch.cuda.is_available() else "cpu")
    print('Device available on this machine: ', device)

    checkpoint_model_transfer = load_model_transfer_checkpoint(in_args.check_point)
    #print(checkpoint_model_transfer.type)
    
    #processed_img = process_image(in_args.path)
    #print(processed_img.shape)
    #imshow(processed_img)

    probs_classes = predict(in_args.path, checkpoint_model_transfer, in_args.top_k)
    print(probs_classes)
    
    check_sanity(in_args.path, in_args.category_names, checkpoint_model_transfer)


if __name__ == '__main__':
    main()
    #called if script is executed on its own
