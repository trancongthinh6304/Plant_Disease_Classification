import cv2
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(root_dir, img_size, val_split, test_split, seed):
  class_name = []
  images = []
  labels = []
  X = []

  for index, folder in enumerate(os.listdir(root_dir)):
    class_name.append(folder)
    image_path = os.path.join(root_dir, folder)
    # print(image_path, len(os.listdir(image_path)))
    for image in os.listdir(image_path):
      labels.append(index)
      images.append(os.path.join(root_dir, folder, image))

  for index, path in enumerate(tqdm(images)):
    try:
      X.append(cv2.resize(cv2.imread(path),dsize = (img_size, img_size)))
    except:
      labels.pop(index)

  X = np.array(X)
  y = np.array(labels)
  Y = to_categorical(y, num_classes = len(class_name))

  X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                      test_size = val_split + test_split,
                                                      stratify = Y, shuffle = True, 
                                                      random_state = seed)
  X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, 
                                                  test_size = test_split/(val_split+test_split), 
                                                  stratify = y_test, shuffle = True, 
                                                  random_state = seed)
  del X, y, Y, images, labels

  return X_train, X_test, X_val, y_val, X_test, y_test, class_name