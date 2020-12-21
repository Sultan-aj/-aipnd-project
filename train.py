# Imports here
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import glob
import random
import train_args
import os

import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable


import torchvision.models as models
import time

import seaborn as sns

import numpy as np
import pandas as pd

from PIL import Image
from collections import OrderedDict


import json


def main():
    
    data_dir = 'flowers'
    #train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    parser = train_args.get_args()

    cli_args = parser.parse_args()

    # check for data directory
    if not os.path.isdir(cli_args.data_directory):
        print(f'Data directory {cli_args.data_directory} was not found.')
        exit(1)

    # check for save directory
    if not os.path.isdir(cli_args.save_dir):
        print(f'Directory {cli_args.save_dir} does not exist. Creating...')
        os.makedirs(cli_args.save_dir)

    # load categories
    with open(cli_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)

    # set output to the number of categories
    output_size = len(cat_to_name)
    print(f"Images are labeled with {output_size} categories.")
    
    data_transforms =  {
        'training': transforms.Compose([
            transforms.RandomRotation(35),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'testing': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    
    #Load the datasets with ImageFolder
    image_datasets = {
        'training' : datasets.ImageFolder(cli_args.data_directory, transform=data_transforms['training']),
        'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    #Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=False),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True)
    }
    
    # Make model
    if not cli_args.arch.startswith("vgg") and not cli_args.arch.startswith("densenet"):
        print("Only supporting VGG and DenseNet")
        exit(1)

    print(f"Using a pre-trained {cli_args.arch} network.")
    model = models.__dict__[cli_args.arch](pretrained=True)
    
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)), # First layer
                          ('relu', nn.ReLU()), # Apply activation function
                          ('fc2', nn.Linear(4096, 102)), # Output layer
                          ('output', nn.LogSoftmax(dim=1)) # Apply loss function
                          ]))

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = classifier
    
    
    def train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader):
        
    
        model.train() # Puts model into training mode
        print_every = 40
        steps = 0
        use_gpu = False

        # Check to see whether GPU is available
        if torch.cuda.is_available():
            use_gpu = True
            model.cuda()
        else:
            model.cpu()

        # Iterates through each training pass based on #epochs & GPU/CPU
        for epoch in range(epochs):
            running_loss = 0
            for inputs, labels in iter(training_loader):
                steps += 1

                if use_gpu:
                    inputs = Variable(inputs.float().cuda())
                    labels = Variable(labels.long().cuda()) 
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels) 

                # Forward and backward passes
                optimizer.zero_grad() # zero's out the gradient, otherwise will keep adding
                output = model.forward(inputs) # Forward propogation
                loss = criterion(output, labels) # Calculates loss
                loss.backward() # Calculates gradient
                optimizer.step() # Updates weights based on gradient & learning rate
                running_loss += loss.item()

                if steps % print_every == 0:
                    validation_loss, accuracy = validate(model, criterion, validation_loader)

                    print("Epoch: {}/{} ".format(epoch+1, epochs),
                            "Training Loss: {:.3f} ".format(running_loss/print_every),
                            "Validation Loss: {:.3f} ".format(validation_loss),
                            "Validation Accuracy: {:.3f}".format(accuracy))


    def validate(model, criterion, data_loader):
        model.eval() # Puts model into validation mode
        accuracy = 0
        test_loss = 0

        for inputs, labels in iter(data_loader):
            if torch.cuda.is_available():
                inputs = Variable(inputs.float().cuda(), volatile=True)
                labels = Variable(labels.long().cuda(), volatile=True) 
            else:
                inputs = Variable(inputs, volatile=True)
                labels = Variable(labels, volatile=True)

            output = model.forward(inputs)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output).data 
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        return test_loss/len(data_loader), accuracy/len(data_loader)

    print("Success")
    '''
    epoch = cli_args.epochs,
    state_dict = model.state_dict(),
    optimizer_dict = optimizer.state_dict(),
    classifier = model.classifier,
    class_to_idx =  nn_model.class_to_idx,
    arch = cli_args.arch
    '''
    
    epochs = cli_args.epochs
    learning_rate = cli_args.learning_rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    train(model, epochs, learning_rate, criterion, optimizer, dataloaders['training'], dataloaders['validation'])
    
    
    
    #Save checkpoint 
    model.class_to_idx = image_datasets['training'].class_to_idx
    model.cpu()
    
    torch.save({'arch': cli_args.arch,
                'state_dict': model.state_dict(), # Holds all the weights and biases
                'class_to_idx': model.class_to_idx},
               'checkpoint.pth')
    
    print("Checkpoint saved.")
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)