import matplotlib.pyplot as plt

def plot_history(history):
    fig,ax = plt.subplots(figsize = (8,5))
    ax.plot(history.history['loss'], label = 'train', color = 'blue')
    ax.plot(history.history['val_loss'], label = 'val',color = 'green')
    ax.set_title('model loss', fontsize = 14)
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(loc = 'upper right', fontsize = 13)