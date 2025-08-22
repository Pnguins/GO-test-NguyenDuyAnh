import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

class ImageClassifier:
    """
    A class to load a trained Keras model and perform inference on a single image.
    """
    def __init__(self, model_path, image_size, class_labels):
        """
        Initializes the classifier.
        
        Args:
            model_path (str): Path to the saved .h5 model file.
            image_size (tuple): The target image size (height, width).
            class_labels (list): A list of class names in the correct order.
        """
        self.model_path = model_path
        self.image_size = image_size
        self.class_labels = class_labels
        self.model = self._load_model()

    def _load_model(self):
        """Loads the Keras model from the specified path."""
        try:
            model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _preprocess_image(self, image_path):
        """Loads and preprocesses an image for prediction."""
        try:
            img = image.load_img(image_path, target_size=self.image_size)
            img_array = image.img_to_array(img)
            img_array /= 255.0  
            img_array = np.expand_dims(img_array, axis=0)  
            return img_array
        except FileNotFoundError:
            print(f"Error: The file '{image_path}' was not found.")
            return None

    def predict(self, image_path):
        """
        Predicts the class for a given image.

        Args:
            image_path (str): The path to the image file.
            
        Returns:
            dict: A dictionary with the predicted label and confidence score, or None on error.
        """
        if self.model is None:
            print("Model is not loaded. Cannot predict.")
            return None
            
        processed_image = self._preprocess_image(image_path)
        if processed_image is None:
            return None

        raw_prediction = self.model.predict(processed_image)
        score = raw_prediction[0][0]

        if score > 0.5:
            predicted_index = 1
            confidence = score
        else:
            predicted_index = 0
            confidence = 1 - score
            
        predicted_label = self.class_labels[predicted_index]

        return {
            "predicted_label": predicted_label,
            "confidence_score": float(confidence)
        }

if __name__ == '__main__':
    inference_config = {
        'model_path': '<FINETUNED_MODEL_PATH>',
        'image_size': (224, 224),
        'class_labels': ['cat', 'dog'],
        'image_to_predict': '<IMAGE_URL>' 
    }

    classifier = ImageClassifier(
        model_path=inference_config['model_path'],
        image_size=inference_config['image_size'],
        class_labels=inference_config['class_labels']
    )

    result = classifier.predict(inference_config['image_to_predict'])

    if result:
        print(f"\n--- Prediction Result ---")
        print(f"Image: {inference_config['image_to_predict']}")
        print(f"Predicted Class: '{result['predicted_label']}'")
        print(f"Confidence: {result['confidence_score']:.4f}")