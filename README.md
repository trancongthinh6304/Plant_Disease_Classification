# Plant Disease Classification
A basic Streamlit application that classifies diseases on plants.

Models are trained using mainly Sklearn and Keras.

This project is created mainly for studying and exploring.

## Introduction
This is a simple Streamlit web application that helps users predict diseases based on their photos on plants.
Currently, we are training on 4 diseases on tomatoes leaves, and 3 diseases on grapes leaves.

### Task definition and Data
#### Task
The task is to classify the disease on leaves using simple classification method: Transered Learning.
#### Data
- Apple Dataset:

| Type of plant | Label                  | Numbers of images |
| ------------- | ---------------------- |:-----------------:|
| Apple         | Healthy                |       2183        |
| Apple         | Bacterial Spot         |       2127        |
| Apple         | Late Bright            |       2499        |
| Apple         | Yellow Leaf Curl Virus |       3209        |
| Apple         | Septoria Leaf Spot     |       2362        |
| Grape         | Healthy                |        423        |
| Grape         | Isariopsis Leaf Spot   |        430        |
| Grape         | Black Rot              |        472        |
| Grape         | Esca Black Measles     |        553        |

Datasets' Link:

- [Grapes](https://drive.google.com/file/d/1QYlqQSzT5QNDUw00Y_RcMLRpRaGc-Qn1/view?usp=sharing)
- [Tomatoes](https://drive.google.com/file/d/1rLjq6NAMsOzh4HevIzYh1afhWhRmws-T/view?usp=sharing)

## About Dataset:

- Images are resized to 256x256 and are normalized (by ./255 or to zero-mean depending on the pretrained model)

## Evaluation Metrics:
- This dataset is unbalanced - that means the normal metrics (accuracy) is not approriate.
- Confusion matrix and f1_score are used.
