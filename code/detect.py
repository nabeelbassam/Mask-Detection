from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from hardcoded import model_conf, data_conf
from utils import Utils
from predict import Predictor
import argparse
import warnings

warnings.filterwarnings("ignore")


class Detect:
    @staticmethod
    def data_prep():
        # training data generator
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=15,
                                           width_shift_range=0.3,
                                           height_shift_range=0.3,
                                           zoom_range=0.25,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           samplewise_center=True,
                                           samplewise_std_normalization=True,
                                           fill_mode='nearest')
        # testing, validation data generator
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_data = train_datagen.flow_from_directory(directory=data_conf.train_data_path,
                                                       batch_size=model_conf.BATCH_SIZE,
                                                       class_mode='categorical',
                                                       target_size=model_conf.TARGET_SIZE)
        validation_data = test_datagen.flow_from_directory(data_conf.validation_data_path,
                                                           target_size=model_conf.TARGET_SIZE)
        test_data = test_datagen.flow_from_directory(data_conf.test_data_path, target_size=model_conf.TARGET_SIZE,
                                                     shuffle=False)
        return train_data, validation_data, test_data

    @staticmethod
    def create_model():
        # We are using DenseNet201 pre-trained model and a fully connected top layer for our classification.
        densenet_model = DenseNet201(input_shape=model_conf.TARGET_SIZE + (3,), weights='imagenet', include_top=False)
        densenet_model.trainable = False

        flatten = Flatten()(densenet_model.layers[-1].output)
        fc = Dense(units=512, activation='relu')(flatten)
        dropout = Dropout(0.35)(fc)
        output = Dense(2, activation='softmax')(dropout)
        final_model = Model(inputs=densenet_model.input, outputs=output)
        # print the summary of the final model
        final_model.summary()
        return final_model

    @staticmethod
    def fit_model(model, train, valid):
        # compiling the model using Adam optimizer and Categorical Cross entropy loss function
        learning_rate = optimizers.schedules.PolynomialDecay(model_conf.starter_learning_rate,
                                                             model_conf.decay_steps,
                                                             model_conf.end_learning_rate, power=0.4)

        optimizer = optimizers.Adam(learning_rate=learning_rate)
        loss = CategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=[model_conf.metric])

        my_callbacks = [
            # Early stopping callback
            EarlyStopping(monitor='val_accuracy', min_delta=1e-3, patience=7,
                          mode='auto', restore_best_weights=False, verbose=1),
            # model check point callback
            ModelCheckpoint(filepath=model_conf.model_path, monitor='accuracy', save_best_only=True,
                            save_weights_only=False, mode='auto', save_freq='epoch', verbose=1)
        ]
        # fitting the model
        history = model.fit(train, epochs=model_conf.EPOCHS,
                            steps_per_epoch=len(train),
                            validation_data=valid,
                            callbacks=[my_callbacks],
                            verbose=1)
        # plot the history
        Utils.plot_metrics(history)
        # save the model in models folder
        Utils.save_model(model)
        # plot train, validation accuracy and loss
        Utils.train_valid_plot(history)
        return model

    @staticmethod
    def evaluate_model(test, model_):
        loss, accuracy = model_.evaluate(test)
        print('Test Accuracy: ', round(accuracy * 100, 2))


parser = argparse.ArgumentParser(description='Mask Detection.')
parser.add_argument('--image', help='Path of image.', default=data_conf.example_to_predict_mask)
parser.add_argument('--mode', help='Run mode, options:(predict, train).', default='predict')
args = parser.parse_args()
# read the image path from the command-line
image_path = args.image
# read the mode for the execution from the command-line
mode = args.mode

if mode in "train":
    # plot random samples
    Utils.plot_random_samples()
    # data augmentation and preprocess
    train_set, valid_set, test_set = Detect.data_prep()
    # create the model
    model = Detect.create_model()
    # fit the model
    model = Detect.fit_model(model, train_set, valid_set)
    # evaluate the model using the test set
    Detect.evaluate_model(test_set, model)
    # show some predicted images with the actual and predicted labels
    Utils.show_image_with_label(test_set, model)
    # save model as onnx model to use to for inference
    Utils.save_as_onnx(model)
if mode == "predict":
    predictor = Predictor(image_path)
    # predict the class of the image based on the saved model
    predictor.predict()