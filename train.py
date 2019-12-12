#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
will train a new network on a dataset and save the model as a checkpoint
python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
python train.py data_dir --save_dir save_directory
python train.py data_dir --arch "vgg13"
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
python train.py data_dir 
"""
import argparse #used to get arguments from user 
import os #used to get current working directory 
import torch #used to identify available device CPU or GPU 
import torchvision.transforms as transforms #used in transformations
from torchvision import datasets # used to load the datasets with ImageFolder
import torchvision.models as models #imports models, pretrained VGG16 is used  
import torch.nn as nn #Relu Linear Dropout used in classifier
from collections import OrderedDict #Used to construct classiefier sequence of steps
import torch.optim as optim #SGD 
import torch.nn.functional as F #used for softmax function
import numpy as np #used in train function to initialize tracker  

def get_input_args():
    """
    Retrieves and parses command line arguments provided by the user when
    they run the program from a terminal. This function uses Python's 
    argparse module to create and define these command line arguments 
    If the user fails to provide some or all of the 3 arguments, then the 
    missing arguments.
    1. save directory -- dir where checkpoint is saved with default value 'cur_dir' 
    2. CNN Model Architecure as -- arch with defualt value 'vgg'
    3. Hyperparameter learning_rate  --arch
    4. Hyperparameter hidden_units   --learning_rate 
    5. Hyperparameter epochs --hidden_units 
    5. Hyperparameter epochs --epochs 
    """ 
    #cur_dir will contain the path of where the script is executing 
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    #Create Argument Parser object named parser 
    parser = argparse.ArgumentParser()
    #Argument: path to a folder
    parser.add_argument('--data_dir', type=str, default='flowers/', \
                        help='Directory containing data for training, validation, and testing')
    #Argument: path to a folder
    parser.add_argument('--save_dir', type=str, default='.', \
                        help='Path to directory where checkpoint will be stored')
    #Argument: architecture type 
    parser.add_argument('--arch', type=str, default='vgg16', \
                        help='Neural network architecure, vgg16, alexnet, densenet161, mobilenet_v2')
    #Argument: hyperparameter learning_rate
    parser.add_argument('--learning_rate', type=float, default='.02', \
                        help='Learning rate for the neural network')
    #Argument: hyperparameter hidden_units
    parser.add_argument('--hidden_units', type=int, default='4096', \
                        help='Number of hidden units in the classifier')
    #Argument: hyperparameter hidden_units
    parser.add_argument('--output_units', type=int, default='102', \
                        help='Number of output units in the classifier')
    #Argument: hyperparameter epochs
    parser.add_argument('--epochs', type=int, default='3', \
                        help='Number of epochs')
    #Assign variables in_args tp parse_args()
    in_args = parser.parse_args()
    #access values of Arguments by printing it 
    #print("Arguments: ", in_args) #DEBUG

    if in_args.hidden_units % 2 != 0:
        raise ValueError('Please choose an even number for hidden units')
    elif in_args.hidden_units < (2*in_args.output_units):
        raise ValueError('Please choose a  number twice the size of output units, default=', in_args.output_units)


    return in_args

def load_loaders(data_dir):
    """load transformed training data, validation data and testing data 
      
    Extended description of function. 
            
    Parameters: 
    arg1 (str): Data director with seperate folders for train valid and test
                          
    Returns: 
    int: return loaders dict 
    """
    #data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #define dataloader parmeters
    batch_size = 32
    num_workers = 0
    
    #normalize the means and standard deveiations 
    #of the images to what the netword expects
    standard_normalization = transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          standard_normalization])
    
    valid_transforms = transforms.Compose([transforms.Resize(256), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          standard_normalization])
    
    test_transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          standard_normalization])
    
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size = batch_size,
                                              num_workers = num_workers,
                                              shuffle = True)
    #print(len(train_loader))#DEBUG
    valid_loader = torch.utils.data.DataLoader(valid_data, 
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle = True)
    #print(len(valid_loader))#DEBUG
    test_loader = torch.utils.data.DataLoader(test_data, 
                                              batch_size = batch_size,
                                              num_workers = num_workers,
                                              shuffle = True)
    #print(len(test_loader))#DEBUG
    loaders= {
            'train': train_loader,
            'valid': valid_loader,
            'test': test_loader
            }
    l_data= {
            'train': train_data,
            'valid': valid_data,
            'test': test_data
            }
    
    return loaders, l_data
    
def load_arch(arch):
    """load pretrained model. 
      
    Parameters: 
    arch (str): desired architecure input from user 
                          
    Returns: 
    dict: Pretrained model 
    """
    if arch == 'vgg16':
        #Implementation Model Architecture
        #Initialized model is saved into model_transfer
        #Model Architecure pre trained VGG16
        model_transfer = models.vgg16(pretrained=True)
        classifier_input = model_transfer.classifier[0].in_features 
        #print('vgg',classifier_input)
    elif arch == 'alexnet':
        model_transfer = models.alexnet(pretrained=True)
        classifier_input = model_transfer.classifier[1].in_features
        #print('alexnet',classifier_input)
    elif arch == 'densenet161':
        model_transfer = models.densenet161(pretrained=True)
        classifier_input = model_transfer.classifier.in_features
        #print('densenet161',classifier_input)
    elif arch == 'mobilenet_v2':
        model_transfer = models.mobilenet_v2(pretrained=True)
        classifier_input = model_transfer.classifier[1].in_features
        #print('mobilenet_v2',classifier_input)

    else:
        raise ValueError('Please only select vgg16, alexnet, densenet161, mobilenet_v2')
    
    #Freez feature parameters, so the net acts as fixed feature extracter 
    #and we only backprop through the new classifer and not the feature extracter
    for param in model_transfer.parameters():
        param.requires_grad = False

    return model_transfer, classifier_input

def create_new_classifier(learning_rate, model_transfer, classifier_input, hidden_units, output_units):
    """Creates a new classifier with 102 ouputs matching flower types. 
      
    Parameters: 
    learning_rate (int):
    model_transfer (int): 

    Returns: 
    str: criterion_transfer
    str: optimizer_transfer
    """
    #Untrained feed-forward network as a classifier, 
    #using ReLU activations and dropout as specified above
    #The classifer follows the original VGG16 model 
    #but modifies the output layer to reflect flower catergories 
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features = classifier_input, out_features = hidden_units, bias = True)),
        ('relu', nn.ReLU(inplace = True)),
        ('dropout', nn.Dropout(p= 0.5, inplace = False)),
        ('fc2', nn.Linear(in_features = hidden_units, out_features = int(hidden_units/2), bias = True)),
        ('relu', nn.ReLU(inplace = True)),
        ('dropout', nn.Dropout(p = 0.5, inplace = False)),
        ('Linear', nn.Linear(in_features =int(hidden_units/2), out_features = output_units, bias = True)),
        ('softmax', nn.LogSoftmax(dim=1)) #this is needed to ouput the prop.
    ]))
    model_transfer.classifier = classifier
    #Spcecifies a loss function and optimizer.
    criterion_transfer = nn.CrossEntropyLoss()
    optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=learning_rate)
    return criterion_transfer, optimizer_transfer


def train(n_epochs, loaders, model, optimizer, criterion, device):
    """Trains and validates classifier portion of the model. 
      
    the train function consits of 1 main loop but two parts, 
    the first is loop goes over batch size of 32
    and is broken on the 10th iteration i.e. 32*10=320 images. 
    at which point the model is validated and metrics are printed
            
    Parameters:
    n_epochs (int): number of epochs 
    loaders (dict): loaders see load_loaders above 
    model (dict): 
    optimizer 
    criterion 
    device (str): GPU or CPU
                          
    Returns: 
    dict: trained model 
    """
    #print('in train',len(loaders['train']))
    """return trained model"""
    #initialize tracker for minimum validation loss
    model.to(device)
    valid_loss_min = np.Inf
    step = 0
    #iterate over epochs
    for epoch in range(1, n_epochs+1):
        #initialize variables to monitor training and validation loss
        valid_loss = 0.0
        train_loss = 0.0
        accuracy = 0.0
        #train the model
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            step += 1 #accumulate steps 
            #move tensors to GPU or CPU depending on device variable
            data, target = data.to(device), target.to(device)
            ##find the loss and update the model parameters accordingly
            
            #initialize weight clearing gradients of optimized variables
            optimizer.zero_grad()
            #forward pass: computing prediction by passing inputs to model and getting log propabilites
            output = model.forward(data)
            #calculate batch loss with log propability and labels
            loss = criterion(output, target)
            #back prop: computing gradient of the loss
            loss.backward()
            #parameter update for single optimization step
            optimizer.step()
            ##record/update the average training loss, using
            train_loss += loss.data 
            
            #COMMENT Print every 10
            #if batch_idx % 100 == 0:
            #print(batch_idx)
            if step % 10 == 0:
                valid_loss = 0.0
                accuracy = 0.0
                
                #validate model
                model.eval()
                for batch_idx, (data, target) in enumerate(loaders['valid']):
                # move to GPU
                    data, target = data.to(device), target.to(device)
                    ## ## TODO: update the average validation loss
                    #forward pass: computing predictions by passing inputs to model
                    output = model.forward(data)
                    #calculate batch loss
                    loss = criterion(output, target)
                    #update average validation loss
                    valid_loss = valid_loss + loss.data 
                    
                    # Class with highest probability is our predicted class, compare with true label 
                    # Calculate accuracy
                    # Model's output is log-softmax, take exponential to get the probabilities
                    # ps = propbality
                    ps = torch.exp(output)
                    # the topk method reutrns the highest k probabilities and the indices of those 
                    #probabilities corresponding to the classes, k = 1
                    # check for equality with labels 
                    equality = (target.data == ps.max(dim=1)[1])
                    # Accuracy is number of correct predictions divided by all predictions; mode 
                    # using the equals we can update our accuracy
                    # once its changed to a float tensor the mean function can be ran
                    accuracy += equality.type(torch.FloatTensor).mean()
        
                # print training/validation statistics 
                #running_loss/print_every takes average of training loss 
                #so everytime its printed the average is takes
                #len of valid_loader tells us how many batches are actually in our test data set 
                #that we are getting from test loader, since we are summing up the batches above
                #we take the total loss and divide it by the number of batches
                #taking the total loss and dividing it by the number of batches 
                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {:.6f}'.format(
                    epoch, 
                    #train_loss/len(train_loader),
                    #valid_loss/len(valid_loader),
                    #accuracy/len(valid_loader)
                    train_loss/len(loaders['train']),
                    valid_loss/len(loaders['valid']),
                    accuracy/len(loaders['valid'])
                ))
                #put the model back in training mode enabling dropout and grads 
                model.train()
                train_loss = 0 #comment        
    #return trained model
    return model

def test(loaders, model, criterion, device):
    """Tests the trained model by measuring its performance on the train dataset 
            
    Parameters:
    loaders (dict): 
    model (dict): 
    criterion ():
    device (str): GPU or CPU
                          
    Returns: 
    NONE 
    """
    #monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    
    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


def save_checkpoint(save_dir, model_transfer, optimizer_transfer, epochs, arch, loaders, l_data):
    """Saves the state of the model that has the new classifier that have been trained. 
      
    Parameters: 
    arg1 (int): Description of arg1 
                          
    Returns: 
    int: Description of return value 
    """
    #print(save_dir, model, data, optimizer, epochs, arch)
    #attach the mapping of classes to indices to the model 
    #as an attribute which makes inference easier later on
    #model_transfer.class_to_idx = train_data.class_to_idx
    #model_transfer.class_to_idx = train_data.class_to_idx
    model_transfer.class_to_idx = l_data['train'].class_to_idx

    checkpoint = {'architecture': 'VGG16',
                  'classifier': model_transfer.classifier,
                  'state_dict': model_transfer.state_dict(),
                  'optimizer_dict': optimizer_transfer.state_dict,
                  #'class_to_idx': train_data.class_to_idx,
                  'class_to_idx': l_data['train'].class_to_idx,
                 }
    torch.save(checkpoint, 'checkpoint_model_transfer.pth')

def main():
    """
    Main 
    """
    #print('python train.py -h')
    in_args = get_input_args()
    print('Arguments', in_args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('device:', device) #DEBUG
    print('Loading Loaders')
    loaders, l_data = load_loaders(in_args.data_dir) 
    #for i in loaders:
    #    print(i, loaders[i])
    #print('len of loaders train: ', len(loaders['train']))
    #print('len of loaders train: ', len(loaders['valid']))
    #print('len of loaders train: ', len(loaders['test']))
    #print('Num test images: ', len(loaders.test_data))
    
    print('Loading Model Architecture') 
    model_transfer, classifier_input = load_arch(in_args.arch)
    #print('Model',model_transfer)#DEBUG

    print('Creating Classifier') 
    criterion_transfer, optimizer_transfer = create_new_classifier(in_args.learning_rate, model_transfer, classifier_input, in_args.hidden_units, in_args.output_units)
    #print('Model',model_transfer)#DEBUG

    print('Training Model, go grab a coffee this will take a while :-)') 
    model_transfer = train(in_args.epochs, loaders, model_transfer, optimizer_transfer, criterion_transfer, device)

    print('Testing Model') 
    test(loaders, model_transfer, criterion_transfer, device)
     
    print('Saving Model') 
    save_checkpoint(in_args.save_dir, model_transfer, optimizer_transfer, in_args.epochs, in_args.arch, loaders, l_data)

if __name__ == '__main__':
    main()
    #called if script is executed on its own 


