import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from hardcoded import model_conf, data_conf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import keras2onnx


class Utils:

    @staticmethod
    def plot_random_samples(mask=True):
        if mask:
            plt.imshow(load_img(data_conf.example_with_mask))
        else:
            plt.imshow(load_img(data_conf.example_without_mask))

    @staticmethod
    def train_valid_plot(history):
        # show train and validation loss
        train_loss = np.array(history.history['loss'])
        val_loss = np.array(history.history['val_loss'])
        plt.semilogy(train_loss, label='Train Loss')
        plt.semilogy(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Loss - Cross Entropy')
        plt.title('Train and Validation Loss')
        plt.show()

        # show train and validation accuracy
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.xlabel('Epoch'),
        plt.ylabel('Accuracy')
        plt.title('Train and Validation Accuracy')
        plt.show()

    @staticmethod
    def save_model(model):
        model.save(model_conf.model_path)

    @staticmethod
    def plot_metrics(history):
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()

    @staticmethod
    def load_image(img_path, show=False):
        # load image from disk to predict
        img = image.load_img(img_path, target_size=model_conf.TARGET_SIZE)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.show()

        return img

    @staticmethod
    def get_model():
        model = load_model(model_conf.model_path, compile=True)
        return model

    @staticmethod
    def show_image_with_label(test, model_):
        image_, label = test.next()
        n_images = 16
        label_names = ['With Mask', 'Without Mask ']
        images = image_[0:n_images, :, :, :]
        labels = label[0:n_images, :]
        predict = np.round(model_.predict(images))

        image_rows = 4
        image_col = int(n_images / image_rows)

        _, axs = plt.subplots(image_rows, image_col, figsize=(30, 20))
        axs = axs.flatten()

        for i in range(n_images):
            img = images[i, :, :, :]
            lab = labels[i, :]
            axs[i].imshow(img)
            pred = predict[i]
            axs[i].axis('off')
            lab, pred = np.argmax(lab), np.argmax(pred)
            axs[i].set_title(label=f'actual label: {label_names[lab]}  |  predicted label: {label_names[pred]}',
                             fontsize=15)
        plt.show()

    @staticmethod
    def save_as_onnx(model):
        onnx_model = keras2onnx.convert_keras(model)
        with open(model_conf.model_onnx, "wb") as f:
            f.write(onnx_model.SerializeToString())
