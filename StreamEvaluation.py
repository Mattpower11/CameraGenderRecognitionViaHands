import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomImageDataset import CustomImageDataset
from PIL import Image


def streamEvaluation(net1:nn.Module, net2:nn.Module, palmar_transforms, dorsal_trasforms, weights_palmar_dorsal:list, palmar_image_path:str, dorsal_image_path:str):
    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the appropriate device
    net1.to(device)
    net2.to(device)
    
    net1.eval()
    net2.eval()

    predicted_label = 0

    with torch.no_grad():

        # Open images using path
        palmar_image = Image.open(palmar_image_path).convert("RGB")
        dorsal_image = Image.open(dorsal_image_path).convert("RGB")

        # Apply transformations
        palmar_image = palmar_transforms(palmar_image)
        palmar_image = palmar_image.unsqueeze(0)
        dorsal_image= dorsal_trasforms(dorsal_image)
        dorsal_image = dorsal_image.unsqueeze(0)

        # Softmax layer
        outputs_alexNetPalmar = net1(palmar_image)
        outputs_alexNetDorsal = net2(dorsal_image)

        # Apply softmax to the outputs
        softmax = torch.nn.Softmax(dim=1)
        probs_alexNetPalmar = softmax(outputs_alexNetPalmar)
        probs_alexNetDorsal = softmax(outputs_alexNetDorsal)

        # Execute the weighted sum
        fused_probs = probs_alexNetPalmar * weights_palmar_dorsal[0] + probs_alexNetDorsal * weights_palmar_dorsal[1]

        # Obtain the predicted class
        predicted_score = torch.max(fused_probs, 1)

        if predicted_score == 0:
            predicted_label = "Male"
        else:
            predicted_label = "Female"

    return predicted_label
