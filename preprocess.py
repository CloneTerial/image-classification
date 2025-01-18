# preprocess.py
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)  # Resize to 32x32
    img_array = image.img_to_array(img)  # Convert to NumPy array
    img_array = img_array / 255.0       # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array