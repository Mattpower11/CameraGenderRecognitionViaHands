import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomImageDataset import CustomImageDataset

def streamEvaluation(net1:nn.Module, net2:nn.Module, transforms:list, weights_palmar_dorsal:list, data_struct:dict, image_path:str, tot_exp: int, batch_size=32):
    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the appropriate device
    net1.to(device)
    net2.to(device)
    
    net1.eval()
    net2.eval()

    tot_labels = torch.tensor([])
    tot_predicted = torch.tensor([])

    with torch.no_grad():
        for exp in range(tot_exp):
            dataset_dorsal = CustomImageDataset(image_dir=image_path, data_structure= data_struct, id_exp=exp, train_test='test', palmar_dorsal='dorsal', transform=transforms)
            data_loader_dorsal = DataLoader(dataset_dorsal, batch_size=batch_size, shuffle=False)
           
            dataset_palmar = CustomImageDataset(image_dir=image_path, data_structure= data_struct, id_exp=exp, train_test='test', palmar_dorsal='palmar', transform=transforms)
            data_loader_palmar= DataLoader(dataset_palmar, batch_size=batch_size, shuffle=False)

            for data_dorsal, data_palmar in zip(data_loader_dorsal, data_loader_palmar):
                
                dorsal_images, labels = data_dorsal
                dorsal_images, labels = dorsal_images.to(device), labels.to(device)

                palmar_images, labels = data_palmar
                palmar_images, labels = palmar_images.to(device), labels.to(device)
                
                # Softmax layer
                outputs_alexNetPalmar = net1(palmar_images)
                outputs_alexNetDorsal = net2(dorsal_images)

                # Apply softmax to the outputs
                softmax = torch.nn.Softmax(dim=1)
                probs_alexNetPalmar = softmax(outputs_alexNetPalmar)
                probs_alexNetDorsal = softmax(outputs_alexNetDorsal)
    
                # Execute the weighted sum
                fused_probs = probs_alexNetPalmar * weights_palmar_dorsal[0] + probs_alexNetDorsal * weights_palmar_dorsal[1]
    
                # Obtain the predicted class
                _, predicted = torch.max(fused_probs, 1)
                
                tot_labels = torch.cat((tot_labels, labels))
                tot_predicted = torch.cat((tot_predicted, predicted))

    return tot_labels, tot_predicted
