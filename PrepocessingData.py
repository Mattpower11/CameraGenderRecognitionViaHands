import cv2
import numpy as np

# To normalize one image [values range 0:1]
def imageNormalization(image: np.ndarray):
    return np.float32(image)/255

# To restore the original pixel scale -> cast on int 
def restoreOriginalPixelValue(image: np.ndarray):
    return (image * 255).astype(np.uint8)

# To pre process a single image 
def preProcessingData(image: np.ndarray, type: str):
    if type == "palmar":
         return preProcessingPalm(image)
    elif type == "dorsal":
        return preProcessingDorsal(image)
    else:
        return None

# To pre process palm image
def preProcessingPalm(image: np.ndarray):
    # Convert the RGB image in GRAYSCALE image
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # Enhancing the constrast to add details 
    contrastImage = cv2.equalizeHist(grayImage)
    # Normalized the image
    normalizedImage = imageNormalization(contrastImage)
    # Image resize for LeNet standard size
    resizedImage = cv2.resize(normalizedImage, (224,224))
    # Convert the grayscale image back to RGB
    rgbImage = cv2.cvtColor(resizedImage, cv2.COLOR_GRAY2RGB)
    # Restore the original image pixel value -> NECESSARIO ORA? 
    finalImage = restoreOriginalPixelValue(rgbImage)
    return finalImage

# To pre process dorsal image
def preProcessingDorsal(image: np.ndarray):
    imageNormalized = imageNormalization(image)
    '''
    Blur the image using cv2.GaussianBlur(image, (3, 3), 0) where (3,3) rappresent the kernel dimension and 0 is the standard deviation for gaussian distribution along the x axis
    cv2.GaussianBlur applies a Gaussian blur to each pixel using a Gaussian function to calculate the average values ​​of the neighboring pixels.
    It estimates by giving more weight to the closest pixels than to the ones further away
    SigmaX = 0 in our function so that OpenCV automatically calculates the standard deviation based on the kernel size
    '''
    blurredImage = cv2.GaussianBlur(imageNormalized, (7, 7), 0) 
    # Image resize for AlexNet standard size
    resizedImage = cv2.resize(blurredImage, (224,224))
    # Restore the original image pixel value -> NECESSARIO ORA? 
    finalImage = restoreOriginalPixelValue(resizedImage)
    return finalImage


