from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# Class to create the dataset used for training and testing
# data_structure: the dictionary containing the data structure
# image_dir: the path to the images
# id_exp: the experiment id
# train_test: the string 'train' or 'test'
# palmar_dorsal: the string 'palmar' or 'dorsal'
# transform: the list of transformations to apply to the images
class CustomImageDataset(Dataset):
    def __init__(self, data_structure, image_dir, id_exp, train_test, palmar_dorsal, transform=None):

        self.labels = {}

        if palmar_dorsal == 'dorsal':    
            self.image_filenames = np.array([riga[1] for riga in data_structure[id_exp][train_test]['images']]).flatten()
        else: 
            self.image_filenames = np.array([riga[0] for riga in data_structure[id_exp][train_test]['images']]).flatten()

        for x in range(0, len(self.image_filenames)):
            self.labels[self.image_filenames[x]] = data_structure[id_exp][train_test]['labels'][x]
        
        self.image_dir = image_dir
        self.palmar_dorsal = palmar_dorsal
        self.transform = transform

    def __len__(self):
        #Returns the number of images in the dataset
        return len(self.image_filenames)

    def __getitem__(self, idx):
        #Returns a tuple (image, label) where image is the image and label is the label
        
        img_name = self.image_filenames[idx]  
        img_path = os.path.join(self.image_dir, img_name)  
        image = Image.open(img_path).convert("RGB")
        label = self.labels[img_name]
        if self.palmar_dorsal == 'palmar':
            image = self.transform[0](image)
        else: 
            image = self.transform[1](image)
        return image, label