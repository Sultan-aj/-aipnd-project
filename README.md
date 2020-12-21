# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Project assets:

Image Classifier Project.ipynb Jupyter Notebook.
Image Classifier Project.html HTML export of the Jupyter Notebook above.
train.py to train a new network on a data set.
predict.py to predict flower name from an image.


# Examples train.py
Help:
```bash
python ./train.py -h
```

Train on **CPU** with default **vgg16**:
```bash
python ./train.py ./flowers/train/
```


# Examples predict.py
Help
```bash
python ./predict.py -h
```

Basic Prediction
```bash
python ./predict.py flowers/valid/5/image_05192.jpg checkpoint.pth
```

Prediction with GPU
```bash
python ./predict.py flowers/valid/5/image_05192.jpg checkpoint.pth --gpu
```
