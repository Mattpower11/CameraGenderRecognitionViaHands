import cv2
import numpy as np
from PIL import Image
from torch import Size 
from torchvision import transforms

from palm_cut import get_palm_cut

# Custom transformation for AlexNet
class CustomAlexNetTransform:
    def __init__(self, applyPalmCut:bool=False):
        self.applyPalmCut = applyPalmCut

    def __call__(self, pil_image):
        # Convert PIL -> RGB -> NumPy
        pil_image = pil_image.convert('RGB')
        np_image = np.array(pil_image, dtype=np.uint8)

        if self.applyPalmCut:
            np_image = get_palm_cut(np_image) 

        # Optional: normalizing to [0..1] before blur
        np_image_norm = imageNormalization(np_image)

        # Blur
        blurred = cv2.GaussianBlur(np_image_norm, (7, 7), 0)

        # Resize to 224×224
        resized = cv2.resize(blurred, (224, 224))

        # Convert back to 0..255
        final_8u = restoreOriginalPixelValue(resized)  # shape: (224, 224, 3)

        # Return PIL image (mode='RGB')
        return Image.fromarray(final_8u, mode='RGB')

# Custom transformation for LeNet
class CustomLeNetTransform:
    def __init__(self, applyPalmCut:bool=False):
        self.applyPalmCut = applyPalmCut

    def __call__(self, pil_image):
        # Convert PIL -> RGB -> NumPy
        pil_image = pil_image.convert('RGB')
        np_image = np.array(pil_image, dtype=np.uint8)

        if self.applyPalmCut:
            np_image = get_palm_cut(np_image) 

        # Convert to GRAY correctly
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        # Equalize hist
        contrast = cv2.equalizeHist(gray_image)

        # Normalize [0..255] -> [0..1]
        norm = imageNormalization(contrast)

        # Resize to 32×32
        resized = cv2.resize(norm, (32, 32))

        # Convert back to [0..255] uint8
        final_8u = restoreOriginalPixelValue(resized)  # shape: (32, 32)

        # Return PIL image (mode='L' = single channel)
        return Image.fromarray(final_8u, mode='L')
    
#Custom transformation for LBP
class CustomLBPTransform:
    def __init__(self, resizeShape:tuple=(1024, 1024), applyPalmCut:bool=False):
        self.applyPalmCut = applyPalmCut
        self.resizeShape = resizeShape

    def __call__(self, image):
        if self.applyPalmCut:
            image = image.convert('RGB')
            image = np.array(image, dtype=np.uint8)
            image = get_palm_cut(image)

        grigio = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        contrasto = cv2.equalizeHist(np.array(grigio))
        resized = cv2.resize(contrasto, self.resizeShape)
        return Image.fromarray(resized, mode='L')
  
class CustomLBPCannyTransform:
    def __init__(self, resizeShape:tuple=(1024, 1024), applyPalmCut:bool=False):
        self.applyPalmCut = applyPalmCut
        self.resizeShape = resizeShape

    def __call__(self, image):
        if self.applyPalmCut:
            image = image.convert('RGB')
            image = np.array(image, dtype=np.uint8)
            image = get_palm_cut(image)

        grigio = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        contrasto = cv2.equalizeHist(np.array(grigio))
        canny = cv2.Canny(contrasto, 50, 200, apertureSize=7, L2gradient = True)
        resized = cv2.resize(canny, self.resizeShape)

        return Image.fromarray(resized, mode='L')
    
# Custom transformation for HOG
class CustomHOGTransform:
    def __init__(self, ksize:Size, sigma:float, resizeShape:tuple=(1024, 1024), applyPalmCut:bool=False):
        self.applyPalmCut = applyPalmCut
        self.resizeShape = resizeShape
        self.ksize=ksize
        self.sigma=sigma

    def __call__(self, pil_image):
        # Convert PIL -> RGB -> NumPy
        image = pil_image.convert('RGB')
        image = np.array(pil_image, dtype=np.uint8)

        if self.applyPalmCut:
            image = get_palm_cut(image)

        gaussian_blurred = cv2.GaussianBlur(image, self.ksize, self.sigma) 
        image = cv2.resize(gaussian_blurred, self.resizeShape)
        return Image.fromarray(image, mode='RGB')

# Custom transformation for HOG
class CustomHOGCannyTransform:
    def __init__(self, ksize:Size, sigma:float, resizeShape:tuple=(1024, 1024), applyPalmCut:bool=False):
        self.applyPalmCut = applyPalmCut
        self.resizeShape = resizeShape
        self.ksize=ksize
        self.sigma=sigma

    def __call__(self, pil_image):

        if self.applyPalmCut:
            pil_image = pil_image.convert('RGB')
            pil_image = np.array(pil_image, dtype=np.uint8)
            pil_image = get_palm_cut(pil_image)

        # Converti in spazio colore LAB
        lab = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2LAB)

        # Separazione dei canali
        l, a, b = cv2.split(lab)

        # Applica CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)

        # Ricombina i canali
        lab_enhanced = cv2.merge((l_enhanced, a, b))

        # Converti di nuovo in RGB
        image_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        grigio = cv2.cvtColor(image_enhanced, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(grigio, 50, 200, apertureSize=7, L2gradient = True)
        image = cv2.resize(canny, self.resizeShape)
        final_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(final_image, mode='RGB')

# To normalize one image [values range 0:1]
def imageNormalization(image: np.ndarray):
    # E.g., convert from [0..255] to [0..1] float
    return image.astype(np.float32) / 255.0

# To restore the original pixel scale -> cast on int 
def restoreOriginalPixelValue(image: np.ndarray):
    # Convert from [0..1] float back to [0..255] uint8
    return (image * 255).astype(np.uint8)

def buildCustomTransform(transform:type, resizeShape:tuple, applyPalmCut:bool=False):
    return transforms.Compose([
        transform(resizeShape=resizeShape, applyPalmCut=applyPalmCut),
        transforms.ToTensor(),          
    ])

def buildCustomTransformHogExtended(transform:type, resizeShape:tuple, ksize:Size, sigma:float, applyPalmCut:bool=False):
    return transforms.Compose([
        transform(resizeShape=resizeShape, applyPalmCut=applyPalmCut, ksize=ksize, sigma=sigma),
        transforms.ToTensor(),          
    ])
