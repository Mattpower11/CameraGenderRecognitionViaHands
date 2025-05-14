import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CustomImageDataset import CustomImageDataset

# Function to train the CNN
# net: the CNN model
# transforms: the list of transformations to apply to the images
# data_struct: the dictionary containing the data structure
# image_path: the path to the images
# palmar_dorsal: the string 'palmar' or 'dorsal'
# tot_exp: the number of experiments
def trainingCNN(net:nn.Module, transforms:list, data_struct:dict, image_path:str, palmar_dorsal:str, tot_exp: int, batch_size=32, weight_decay=5e-05, learning_rate=0.001):
    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_values = []

    for exp in range(tot_exp):
        #Training the model

        dataset_train = CustomImageDataset(image_dir=image_path, data_structure = data_struct, id_exp=exp, train_test='train', palmar_dorsal=palmar_dorsal, transform=transforms)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
        net.train()
        running_loss = 0.0
        for _, data in enumerate(data_loader_train, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss_values.append(running_loss / len(data_loader_train))
        print(f'Epoch {exp + 1}, Loss: {running_loss / len(data_loader_train):.4f}')

    return loss_values

def testCNN(net:nn.Module, transforms:list, data_struct:dict, image_path:str, palmar_dorsal:str, tot_exp: int, batch_size=32):
    # Move the model to the appropriate device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    tot_labels = torch.tensor([])
    tot_predicted = torch.tensor([])
    with torch.no_grad():

        for exp in range(tot_exp):

            dataset_test = CustomImageDataset(image_dir=image_path, data_structure= data_struct, id_exp=exp, train_test='test', palmar_dorsal=palmar_dorsal, transform=transforms)
            data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            for data in data_loader_test:
                
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                # Softmax layer
                outputs = net(images)

                # Classification layer
                _, predicted = torch.max(outputs.data, 1)

                tot_labels = torch.cat((tot_labels, labels))
                tot_predicted = np.concatenate((tot_predicted, predicted))

    return tot_labels, tot_predicted