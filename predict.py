import cv2
import numpy as np

def predict(class_name, model, img_path, img_size):
    X = []
    X.append(cv2.resize(cv2.imread(img_path),dsize = (img_size, img_size)))
    X = np.array(X)
    pred = model.predict(X)
    return class_name[np.argmax(pred)], pred[np.argmax(pred)]