import torch.nn as nn
from CNNTrainingTest import testCNN, trainingCNN
from MyLeNetCNN import MyLeNetCNN
from PrepareData import prepare_data
import torchvision
from PerformanceEvaluation import *
from StreamEvaluation import streamEvaluation
from CustomTransform import buildAlexNetTransformations, buildLeNetTransformations


# Set number of experiments
num_exp = 50
image_path = 'path/to/dataset'
csv_path = 'GenderRecognitionViaHands/11kHands_gender_afifi2017_/Progetto/HandInfo.csv'
num_train = 400
num_test = 200

# Create the networks
leNet = MyLeNetCNN(num_classes=2)
alexNet1 = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
alexNet2 = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)

# Customize AlexNet1
# Update the final layer to output 2 classes
num_features = alexNet1.classifier[6].in_features
alexNet1.classifier[6] = nn.Linear(num_features, 2)

# Freeze all layers except the newly added fully connected layer
for param in alexNet1.parameters():
    param.requires_grad = False
for param in alexNet1.classifier[6].parameters():
    param.requires_grad = True

# Customize AlexNet2
# Update the final layer to output 2 classes
num_features = alexNet2.classifier[6].in_features
alexNet2.classifier[6] = nn.Linear(num_features, 2)

# Freeze all layers except the newly added fully connected layer
for param in alexNet2.parameters():
    param.requires_grad = False
for param in alexNet2.classifier[6].parameters():
    param.requires_grad = True

# Set the networks
net_palmar = leNet
net_dorsal = alexNet2

weight_palmar = 0.4
weight_dorsal = 0.6

# Build the tranformations for the networks
palmar_transforms = buildAlexNetTransformations()
if isinstance(net_palmar, MyLeNetCNN):
        palmar_transforms = buildLeNetTransformations()
elif isinstance(net_palmar, torchvision.models.AlexNet):
        palmar_transforms = buildAlexNetTransformations()

dorsal_transforms = buildAlexNetTransformations()
if isinstance(net_dorsal, MyLeNetCNN):
        dorsal_transforms = buildLeNetTransformations()
elif isinstance(net_dorsal, torchvision.models.AlexNet):
        dorsal_transforms = buildAlexNetTransformations()

transforms = [
    palmar_transforms,
    dorsal_transforms
]

# Weights for the fusion
weights_palmar_dorsal = [weight_palmar, weight_dorsal]

# Prepare data
data_struct = prepare_data(csv_path=csv_path, num_exp=num_exp, num_train=num_train, num_test=num_test)

# Training the networks
print('Begin Palm Training\n')
train_loss_p = trainingCNN(net=net_palmar, transforms=transforms, data_struct=data_struct, image_path=image_path, palmar_dorsal='palmar', tot_exp=num_exp)
print('\nFinished Palm Training\n')
print('Begin Dorsal Training\n')
train_loss_d = trainingCNN(net=net_dorsal, transforms=transforms, data_struct=data_struct, image_path=image_path, palmar_dorsal='dorsal', tot_exp=num_exp)
print('\nFinished Dorsal Training\n')

# Test the networks
print('Begin Palm Testing')
palmar_labels, palmar_predicted = testCNN(net=net_palmar, transforms=transforms, data_struct=data_struct, image_path=image_path, palmar_dorsal='palmar', tot_exp=num_exp)
print('Finished Palm Testing\n')
print('Begin Dorsal Testing')
dorsal_labels, dorsal_predicted = testCNN(net=net_dorsal, transforms=transforms, data_struct=data_struct, image_path=image_path, palmar_dorsal='dorsal', tot_exp=num_exp)
print('Finished Dorsal Testing\n')

# Evaluate the unified network
print("Begin Unified Network Testing")
un_labels, un_predicted  = streamEvaluation(net1=net_palmar, net2=net_dorsal, transforms=transforms, weights_palmar_dorsal=weights_palmar_dorsal, data_struct=data_struct, image_path=image_path, tot_exp=num_exp)
print("Finished Unified Network Testing\n")


# Performance evaluation
calculate_confusion_matrix(palmar_labels, palmar_predicted)
calculate_confusion_matrix(dorsal_labels, dorsal_predicted)
calculate_confusion_matrix(un_labels, un_predicted)

# Calculate the loss plot
calculate_loss_plot(train_loss_p)
calculate_loss_plot(train_loss_d)

# Print the performance metrics
print("\nPerformance Metrics\n")

print(f"\nPalmar Network= {type(net_palmar).__name__}")
print(f"Dorsal Network= {type(net_dorsal).__name__}\n")

print("\nAccuracy Palmar Network: ", calculate_accuracy(palmar_labels, palmar_predicted))
print("Precision Palmar Network: ", calculate_precision(palmar_labels, palmar_predicted))
print("Recall Palmar Network: ", calculate_recall(palmar_labels, palmar_predicted))
print("F1 Score Palmar neNetworkt: ", calculate_f1_score(palmar_labels, palmar_predicted),"\n")

print("\nAccuracy Dorsal Network: ", calculate_accuracy(dorsal_labels, dorsal_predicted))
print("Precision Dorsal Network: ", calculate_precision(dorsal_labels, dorsal_predicted))
print("Recall Dorsal Network: ", calculate_recall(dorsal_labels, dorsal_predicted))
print("F1 Score Dorsal Network: ", calculate_f1_score(dorsal_labels, dorsal_predicted),"\n")

print("\nAccuracy Unified Network: ", calculate_accuracy(un_labels, un_predicted))
print("Precision Unified Network: ", calculate_precision(un_labels, un_predicted))
print("Recall Unified Network: ", calculate_recall(un_labels, un_predicted))
print("F1 Score Unified Network: ", calculate_f1_score(un_labels, un_predicted),"\n")