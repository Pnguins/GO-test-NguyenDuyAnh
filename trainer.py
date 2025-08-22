import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

class VGG16FineTuner:
    """
    A class to handle the fine-tuning of a VGG16 model for binary image classification.
    """
    def __init__(self, config):
        """
        Initializes the trainer with a configuration dictionary.
        
        Args:
            config (dict): A dictionary containing training parameters.
        """
        self.config = config
        self.model = None
        self.history = None
        self.train_generator = None
        self.validation_generator = None

    def _build_model(self):
        """Builds and compiles the VGG16-based model."""
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.config['image_size'] + (3,)
        )
        for layer in base_model.layers:
            layer.trainable = False

        self.model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Model built and compiled successfully.")
        self.model.summary()

    def _prepare_data_generators(self):
        """Prepares the training and validation data generators."""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            self.config['train_dir'],
            target_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            class_mode='binary'
        )

        self.validation_generator = validation_datagen.flow_from_directory(
            self.config['validation_dir'],
            target_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            class_mode='binary',
            shuffle=False  # Crucial for confusion matrix
        )
        print("Data generators prepared.")

    def train(self):
        """Orchestrates the model building, data preparation, and training process."""
        self._build_model()
        self._prepare_data_generators()

        print("Starting model training...")
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.config['batch_size']
        )
        print("Training finished.")
        self.save_model()

    def save_model(self):
        """Saves the trained model to the path specified in the config."""
        self.model.save(self.config['model_save_path'])
        print(f"Model saved to {self.config['model_save_path']}")

    def plot_and_save_history(self):
        """Plots the training history and saves it to a file."""
        if not self.history:
            print("No training history found. Please run train() first.")
            return

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs_range = range(self.config['epochs'])

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
        # Save the figure
        plt.savefig(self.config['history_plot_path'])
        print(f"Training history plot saved to {self.config['history_plot_path']}")
        plt.show()

    def generate_and_save_confusion_matrix(self):
        """Generates and saves a confusion matrix for the validation set."""
        if self.model is None or self.validation_generator is None:
            print("Model or validation generator not available. Run train() first.")
            return

        # Get the true labels
        y_true = self.validation_generator.classes
        
        # Get the predicted probabilities
        y_pred_probs = self.model.predict(self.validation_generator, verbose=1)
        
        # Convert probabilities to class labels (0 or 1)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        # Ensure lengths match
        if len(y_pred) != len(y_true):
            print("Warning: Prediction and true label counts do not match.")
            print(f"Trimming true labels to match prediction count: {len(y_pred)}")
            y_true = y_true[:len(y_pred)]

        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class labels from the generator
        class_names = list(self.validation_generator.class_indices.keys())

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save the figure
        plt.savefig(self.config['cm_plot_path'])
        print(f"Confusion matrix plot saved to {self.config['cm_plot_path']}")
        plt.show()


if __name__ == '__main__':
    # Fine tuning config
    training_config = {
        'image_size': (224, 224),
        'batch_size': 32,
        'epochs': 25,
        'learning_rate': 0.0001,
        'train_dir': '<PATH_TO_TRAIN_FOLDER>',
        'validation_dir': '<PATH_TO_TEST_FOLDER',
        'model_save_path': 'output/vgg16_finetuned_2_class_oop.h5', # The model name and save location change if needed 
        'history_plot_path': 'output/training_history.png',
        'cm_plot_path': 'output/confusion_matrix.png'     
    }

    tuner = VGG16FineTuner(config=training_config)
    tuner.train()
    
    # Generate and save plots
    tuner.plot_and_save_history()
    tuner.generate_and_save_confusion_matrix()