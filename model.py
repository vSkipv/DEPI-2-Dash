import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

class SkinCancerModel:
    def __init__(self):
        self.model = None
        self.class_names = ['BCC', 'SCC', 'MEL', 'NEV', 'ACK', 'SEK']
        self.model_path = 'models/skin_cancer_model.h5'
        
    def create_model(self):
        """
        Create a simple CNN model for skin cancer detection
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

    def load_model(self, model_path=None):
        """
        Load the trained model from the specified path
        """
        if model_path:
            self.model_path = model_path
            
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                return True
            else:
                print(f"Model file not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def save_model(self):
        """
        Save the trained model
        """
        if self.model:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            return True
        return False

    def preprocess_image(self, image_path):
        """
        Preprocess the image for model input
        """
        try:
            # Read and preprocess the image
            img = Image.open(image_path)
            img = img.resize((224, 224))  # Resize to model's expected input size
            img_array = np.array(img)
            
            # Convert to RGB if grayscale
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Normalize pixel values
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def predict(self, image_path):
        """
        Make prediction on the input image
        """
        if self.model is None:
            print("Model not loaded. Creating new model...")
            self.create_model()
            if not self.load_model():
                return None, 0.0

        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return None, 0.0

            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            return self.class_names[predicted_class], confidence

        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None, 0.0

    def train(self, train_data, validation_data, epochs=10):
        """
        Train the model with provided data
        """
        if self.model is None:
            self.create_model()
            
        try:
            history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs
            )
            self.save_model()
            return history
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None

# Example usage:
if __name__ == "__main__":
    model = SkinCancerModel()
    
    # Create and save a new model
    model.create_model()
    model.save_model()
    
    # Example prediction
    # prediction, confidence = model.predict("path_to_image.jpg")
    # print(f"Prediction: {prediction}, Confidence: {confidence}") 