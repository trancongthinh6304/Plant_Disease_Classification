import sys
sys.path.append(r'C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0\LocalCache\local-packages\Python38\site-packages')

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from PIL import Image, ImageOps
from keras.preprocessing import image
import io
import time

model = load_model(r'C:\Users\Administrator\Downloads\weights-24.hdf5')
class_name = ['Bệnh đốm vi khuẩn','Không có bệnh','Bệnh xoăn vàng lá','Bệnh úa muộn (Bệnh sương mai)','Bệnh đốm lá cà chua (Septoria)']

import streamlit as st
st.markdown("<h3 style='text-align: center; color: black;'> Vòng chung kết cấp Thành phố cuộc thi KHKT dành cho học sinh trung học năm học 2020-2021 </h3>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: red;'>Phần mềm ứng dụng Trí Tuệ Nhân Tạo hỗ trợ nhận diện và chẩn đoán bệnh ở cây trồng tại nhà</h1>", unsafe_allow_html=True)
st.write("21-1969")
file = st.file_uploader("Hãy tải lên một ảnh:", type=["jpg","png","jpeg"])

if file is None:
	st.text("Hãy tải hình ảnh lên")
else:
	image = Image.open(file).convert("RGB")
	st.image(image, caption='Ảnh đã tải lên', width = 200)
	img_array = np.array(image)
	X = []
	X.append(cv2.resize(img_array,dsize=(256,256)))
	X = np.array(X)
	predictions = model.predict(X)
	latest_iteration = st.empty()
	bar = st.progress(0)

	for i in range(100):
		latest_iteration.text(f'Đang xử lý ({i+1}%)')
		bar.progress(i + 1)
		time.sleep(0.01)

	st.write("Đây có thể là bệnh: ", class_name[np.argmax(predictions)])
