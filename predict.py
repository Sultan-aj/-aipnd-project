import json
import torch
import predict_args
import warnings

from collections import OrderedDict
from torch import nn, optim
from PIL import Image
from torchvision import models
from torchvision import transforms

parser = predict_args.get_args()

cli_args = parser.parse_args()

# load categories
with open(cli_args.categories_json, 'r') as f:
     cat_to_name = json.load(f)
        

def main():
    # load the cli args
    parser = predict_args.get_args()

    cli_args = parser.parse_args()

    # Start with CPU
    device = torch.device("cpu")

    # Requested GPU
    if cli_args.use_gpu:
        device = torch.device("cuda:0")

    
   
    # load the saved model
    model = load_model(cli_args.checkpoint_file)
    #print(model)
    
    
    
    # Run the prediction
    predict(cli_args.path_to_image, model, topk=5)

    top_probs, top_labels, top_flowers = predict(cli_args.path_to_image, model)
    print(top_probs)
    top_probs = top_probs[0].detach().numpy() #converts from tensor to nparray
    print(top_probs)

    
    flower_num = cli_args.path_to_image.split('/')[2]
    title_ = cat_to_name[flower_num] # Calls dictionary for name        

    
    label = top_labels[0]

    prob = top_probs[0]
    print(prob)
    # display the results
    print(f'Parameters\n---------------------------------')

    print(f'Image  : {cli_args.path_to_image}')
    print(f'Model  : {cli_args.checkpoint_file}')
    print(f'Device : {device}')

    print(f'\nPrediction\n---------------------------------')

    print(f'Flower      : {cat_to_name[flower_num]}')
    print(f'Label       : {label}')
    print(f'Probability : {prob*100:.2f}')


    
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

    # Process a PIL image for use in a PyTorch model
    img = Image.open(image).convert("RGB")
    adjust = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])
    img_tensor = adjust(img)

    return img_tensor
           
   
        
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.  '''

    # TODO: Implement the code to predict the class from an image file

    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    #probs = torch.exp(model.forward(processed_image))
    probs = model.forward(processed_image)
    top_probs, top_labs = probs.topk(topk)
    
    top_probs = top_probs.exp()

    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labs = top_labs[0].numpy()

    top_labels = []
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]

    return top_probs, top_labels , top_flowers
        
def load_model( checkpoint_path = 'checkpoint.pth'):
    """
        Loads model checkpoint saved by train.py
        """
    
   
    checkpoint = torch.load(checkpoint_path)
    
    model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    #model.load_state_dict(checkpoint['state_dict'])
    state_dict = checkpoint['state_dict']
    new_state_dict= OrderedDict()
    for k, v in state_dict.items():
        name=k[7:] #remove 'module.' of DataParallel
        new_state_dict[name]=v

    model.load_state_dict(new_state_dict, strict=False)
    
    return model
        
    
    
if __name__ == '__main__':
    # some models return deprecation warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()


    