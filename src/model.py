# %% Packages

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.regularizers import l2

# %% Classes


class OxfordFlower102Model:
    """
    This class is initializing the model
    """

    def __init__(self, config):
        self.config = config
        self.base_model = self.build_model()
        tf.random.set_seed(self.config.model.random_seed)

    def build_model(self):
        """
        This method build the basic model. The basic model describes the pre-trained model plus a dense layer
        on top which is individualized to the number of categories needed. The model is also compiled
        :return: A compiled tensorflow model
        """
        pre_trained_model = self.initialize_pre_trained_model()
        top_model = self.create_top_layers()

        top1err = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top1')
        top2err = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2')
        top5err = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')

        model = Sequential()
        model.add(pre_trained_model)
        model.add(top_model)

        sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

        #model.compile(
        #    loss=self.config.model.loss,
        #    metrics=["accuracy", top1err, top2err, top5err],
        #    optimizer=tf.keras.optimizers.Adam(
        #        learning_rate=self.config.model.learning_rate
        #    ),
        #)
        model.compile(
            loss=self.config.model.loss,
            metrics=["accuracy", top1err, top2err, top5err],
            optimizer=sgd,
        )
        model.summary()
        return model

    def unfreeze_top_n_layers(self, model):
        """
        This method unfreezes a certain number of layers of the pre-trained model and combines it subsequently with the
        pre-trained top layer which was added within the 'create_top_layers' method and trained within the 'build_model'
        class
        :param model: Tensorflow model which was already fitted
        :param ratio: Float of how many layers should not be trained of the entire model
        :return: Compiled tensorflow model
        """
        base_model = model.layers[0]
        trained_top_model = model.layers[1]

        base_model.trainable = True
        for layer in base_model.layers[:-2]:
            layer.trainable = False

        top1err = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top1')
        top2err = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2')
        top5err = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')

        fine_tune_model = Sequential()
        fine_tune_model.add(base_model)
        fine_tune_model.add(trained_top_model)

        adjusted_learning_rate = (
            self.config.model.learning_rate / self.config.model.learning_rate_shrinker
        )
        fine_tune_model.compile(
            loss=self.config.model.loss,
            metrics=["accuracy", top1err, top2err, top5err],
            optimizer=tf.keras.optimizers.Adam(learning_rate=adjusted_learning_rate),
        )
        fine_tune_model.summary()
        return fine_tune_model

    def initialize_pre_trained_model(self):
        """
        This method calls the pre-trained model. In this case we are loading the MobileNetV2
        :return: Tensorflow model
        """


        image_shape = (
            self.config.data_loader.target_size,
            self.config.data_loader.target_size,
            3,
        )
        base_model = ResNet50(
            input_shape=image_shape, include_top=False, weights='imagenet'
        )
        base_model.trainable = False
        return base_model

    def create_top_layers(self):
        """
        Creating the tensorflow top-layer of a model
        :return: Tensorflow Sequential model
        """
        top_model = Sequential()
        top_model.add(GlobalAveragePooling2D())
        #top_model.add(Flatten())
        top_model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
        top_model.add(Dropout(rate=0.4))
        top_model.add(Dense(self.config.model.number_of_categories, activation="softmax"))
        return top_model
