# %% Packages

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# %% Classes


class OxfordFlower102Trainer:
    """
    This class is training the base-model and fine-tunes the model
    """

    def __init__(self, model, data_generator, config):
        self.config = config
        self.model = model
        self.train_data_generator = data_generator.train_generator
        self.val_data_generator = data_generator.val_generator
        self.test_data_generator = data_generator.test_generator
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

        self._init_callbacks()
        print("Train the base Model!")
        self.train_model()
        #print("Fine Tune the model")
        #self.train_fine_tune()
        print("Evaluate the Model!")
        self.evaluate(self.test_data_generator)
        print("Save the Model!")
        self.save_model()

    def evaluate(self, testSet):
        testingModel = tf.keras.models.load_model('final_model.keras')
        evaluation = testingModel.evaluate(testSet)
        print(evaluation)
        print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

    def _init_callbacks(self):
        self.custom_callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                    ),
                ModelCheckpoint(
                    'final_model.keras',
                    save_best_only=True,
                    monitor='val_loss')
                ]

    def train_model(self):
        """
        This method is training the base_model
        :return: /
        """
        self.final_model = self.model.base_model
        history = self.final_model.fit(
                self.train_data_generator,
                verbose=self.config.trainer.verbose_training,
                epochs=self.config.trainer.number_of_base_epochs,
                validation_data=self.val_data_generator,
                callbacks=self.custom_callbacks,
                )
        self.append_model_data(history)
        self.plot_history("model")
    
    def train_fine_tune(self):
        """
        This method is unfreezing some layers of the already trained model and re-trains the model
        :return: /
        """
        total_epochs = (
            self.config.trainer.number_of_base_epochs
            + self.config.trainer.number_of_fine_tune_epochs
        )
        self.fine_tune_model = self.model.unfreeze_top_n_layers(
            self.model.base_model
        )

        fine_tune_history = self.fine_tune_model.fit(
            self.train_data_generator,
            verbose=self.config.trainer.verbose_training,
            initial_epoch=self.config.trainer.number_of_base_epochs,
            epochs=total_epochs,
            validation_data=self.val_data_generator,
            callbacks=self.custom_callbacks,
        )
        self.append_model_data(fine_tune_history)
        self.plot_history("fine_tune_model")
    

    def append_model_data(self, history):
        """
        This method is
        :param history: Tensorflow model history
        :return: /
        """
        self.loss.extend(history.history["loss"])
        self.val_loss.extend(history.history["val_loss"])

        self.acc.extend(history.history["accuracy"])
        self.val_acc.extend(history.history["val_accuracy"])

    def plot_history(self, title):
        """
        This method is plotting the accuracy and loss of the plots
        :param title: str - Used to save the png
        :return: /
        """
        fig, axs = plt.subplots(figsize=(10, 5), ncols=2)
        axs = axs.ravel()
        axs[0].plot(self.loss, label="Training")
        axs[0].plot(self.val_loss, label="Validation")
        axs[0].set_title("Loss")
        axs[0].axvline(
                x=(self.config.trainer.number_of_base_epochs - 1),
                ymin=0,
                ymax=1,
                label="BaseEpochs",
                color="green",
                linestyle="--",
                )
        axs[0].legend()

        axs[1].plot(self.acc, label="Training")
        axs[1].plot(self.val_acc, label="Validation")
        axs[1].set_title("Accuracy")
        axs[1].axvline(
                x=(self.config.trainer.number_of_base_epochs - 1),
                ymin=0,
                ymax=1,
                label="BaseEpochs",
                color="green",
                linestyle="--",
                )
        axs[1].legend()

        fig.savefig(f"./reports/figures/history_{title}.png")

    def save_model(self):
        """
        Saving the fine-tuned model
        :return: /
        """
        path = "./models/oxford_flower102_fine_tuning.keras"
        self.final_model.save(filepath=path)
