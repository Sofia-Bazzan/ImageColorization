#import required packages
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import trange
import os
from PIL import Image
import tensorflow as tf

#Dataset Creation and preprocessing
np.random.seed(0)


DATA_DIR = "flowers"
x_paths = [DATA_DIR+"/"+i for i in sorted(os.listdir(DATA_DIR)) if not i.startswith(".")]
data = []

#resize images if necessary
for image_path_i in trange(len(x_paths)):
    image_path = x_paths[image_path_i]

    try:
        image_array = np.array(Image.open(image_path))
        if len(image_array.shape) == 3:  # Check if the image has 3 dimensions
            data.append(tf.image.resize(image_array, (128, 128)))
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

#create the splits
data = np.array(data)
print(data.shape)
print("train/test/val split pt1")
X_train, X_test = train_test_split(data, test_size=0.2, random_state=1)
print("train/test/val split pt2")
X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

#save array into folder
if not os.path.exists('dataset'):
    os.makedirs('dataset')

with open('dataset/X_train.npy', 'wb+') as f:
    np.save(f, np.array(X_train))

with open('dataset/X_val.npy', 'wb+') as f:
    np.save(f, np.array(X_val))

with open('dataset/X_test.npy', 'wb+') as f:
    np.save(f, np.array(X_test))
