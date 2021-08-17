from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

def create_callbacks(save_dir, es_patience, min_lr, lr_patience, lr_factor):
    filepath = save_dir + "/weights-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_accuracy', verbose = 1, 
                                 save_best_only = False, mode = 'max')

    monitor = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                               min_delta = 0.0001, 
                                               patience = es_patience, 
                                               verbose = 1, 
                                               mode = 'min',
                                               restore_best_weights = True)

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss",
                                                        factor = lr_factor,
                                                        patience = lr_patience,
                                                        verbose = 1,
                                                        mode = 'min',
                                                        min_delta = 0.0001,
                                                        cooldown = 0,
                                                        min_lr = min_lr)

    callbacks_list = [checkpoint, monitor, lr_scheduler]
    return callbacks_list