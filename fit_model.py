from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(rotation_range = 25, width_shift_range = 0.1,
                         height_shift_range = 0.1, shear_range = 0.2, 
                         zoom_range = 0.2, horizontal_flip = True, 
                         fill_mode = "nearest", rescale = 1./255)

aug_val = ImageDataGenerator(rescale=1./255)

def fit_model(model, batch_size, num_epoch, X_train, y_train, X_val, y_val, callbacks_list):
    history = model.fit(aug.flow(X_train, y_train, batch_size = batch_size),
                                 epochs = num_epoch, steps_per_epoch = len(X_train)//batch_size,
                                 validation_data = aug_val.flow(X_val, y_val,
                                 batch_size = batch_size), callbacks = callbacks_list)
    return history, model