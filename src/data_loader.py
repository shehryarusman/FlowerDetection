# %% Packages

import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

# %% Classes


class OxfordFlower102DataLoader:
    """
    This class loads the images and labels and embeds them into ImageDataGenerators.
    """

    def __init__(self, config):
        self.config = config
        (
            self.train_generator,
            self.val_generator,
            self.test_generator,
        ) = self.create_generators()

    def create_generators(self):
        """
        This method loads the labels and images, which are already split into train, test and validation.
        Furthermore, we add an additional step to the preprocessing function, which is required for the pre-trained
        model. Afterwards we create ImageGenerators from tensorflow for train, test and validation.
        :return: ImageDataGenerator for training, validation and testing
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self._image_and_labels()
        train_augment_settings, test_augment_settings = self._add_preprocess_function()

        # Data Augmentation setup initialization
        train_data_gen = ImageDataGenerator(**train_augment_settings)
        valid_data_gen = ImageDataGenerator(**test_augment_settings)
        test_data_gen = ImageDataGenerator(**test_augment_settings)

        # Setting up the generators
        training_generator = train_data_gen.flow(
            x=X_train, y=y_train, batch_size=self.config.data_loader.batch_size
        )
        validation_generator = valid_data_gen.flow(
            x=X_val, y=y_val, batch_size=self.config.data_loader.batch_size
        )
        test_generator = test_data_gen.flow(
            x=X_test, y=y_test, batch_size=self.config.data_loader.batch_size
        )
        return training_generator, validation_generator, test_generator

    def _add_preprocess_function(self):
        """
        This function adds the pre-processing function for the MobileNet_v2 to the settings dictionary.
        The pre-processing function is needed since the base-model was trained using it.
        :return: Dictionaries with multiple items of image augmentation
        """
        train_augment_settings = self.config.data_loader.train_augmentation_settings
        test_augment_settings = self.config.data_loader.test_augmentation_settings
        train_augment_settings.update(
            {
                "preprocessing_function": tf.keras.applications.resnet50.preprocess_input
            }
        )
        test_augment_settings.update(
            {
                "preprocessing_function": tf.keras.applications.resnet50.preprocess_input
            }
        )
        return train_augment_settings, test_augment_settings

    def _image_and_labels(self):
        splits = ['train+validation', 'test[90%:]', 'test[:90%]']
        datasets, info = tfds.load(name='oxford_flowers102', split=splits, as_supervised=True, with_info=True)
        train_dataset, val_dataset, test_dataset = datasets

        trainY = tf.stack([y for x, y in tfds.as_numpy(train_dataset)], axis=0)
        valY = tf.stack([y for x, y in tfds.as_numpy(val_dataset)], axis=0)
        testY = tf.stack([y for x, y in tfds.as_numpy(test_dataset)], axis=0)

        trainX = np.array([tf.image.resize(image, (self.config.data_loader.target_size, self.config.data_loader.target_size)).numpy() for image, _ in tfds.as_numpy(train_dataset)])
        valX = np.array([tf.image.resize(image, (self.config.data_loader.target_size, self.config.data_loader.target_size)).numpy() for image, _ in tfds.as_numpy(val_dataset)])
        testX = np.array([tf.image.resize(image, (self.config.data_loader.target_size, self.config.data_loader.target_size)).numpy() for image, _ in tfds.as_numpy(test_dataset)])

        return trainX, valX, testX, trainY, valY, testY

    '''
    def _image_and_labels(self):
        """
        This method loads labels and images and afterwards split them into training, validation and testing set
        :return: Trainings, Validation and Testing Images and Labels
        """
        y = self._load_labels()
        X = self._loading_images_array()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=self.config.data_loader.train_size,
            random_state=self.config.data_loader.random_state,
            shuffle=True,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            train_size=self.config.data_loader.train_size,
            random_state=self.config.data_loader.random_state,
            shuffle=True,
            stratify=y_train,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _load_labels(self):
        """
        Fetches labels from the TFDS Oxford Flowers 102 dataset, and one-hot encodes them.
        :return: Numpy array of one-hot encoding labels
        """
        dataset, _ = tfds.load('oxford_flowers102', split='train', as_supervised=True, with_info=True)
        label_list = [label for _, label in tfds.as_numpy(dataset)]
        label_2d = np.array(label_list).reshape(-1, 1)

        encoder = OneHotEncoder()
        one_hot_labels = encoder.fit_transform(label_2d).toarray()
        return one_hot_labels

    def _loading_images_array(self):
        """
        Fetches images from the TFDS Oxford Flowers 102 dataset, resizes them, and converts to a numpy array.
        :return: Numpy array of the images
        """
        dataset, _ = tfds.load('oxford_flowers102', split='train', as_supervised=True, with_info=True)
        image_list = []
        for image, _ in tfds.as_numpy(dataset):
            resized_image = tf.image.resize(image, (self.config.data_loader.target_size, self.config.data_loader.target_size))
            resized_image = resized_image.numpy()  # Converting tensor to numpy array
            image_list.append(resized_image)
        return np.array(image_list)
        '''
