import ee

class CropStressModel:
    def __init__(self):
        self.feature_names = ['NDVI', 'NDWI', 'NDMI', 'MSI', 'NDRE']
        self.classifier = None

    def train(self):
        """
        Trains a Random Forest classifier on GEE using synthetic training data.
        In production, this would load a FeatureCollection of ground truth points.
        """
        print("Training GEE-native Random Forest model...")
        
        # Synthetic Training Data (FeatureCollection)
        # 0: Healthy, 1: Water Stress, 2: Heat Stress, 3: Nutrient, 4: Disease
        training_features = ee.FeatureCollection([
            # Healthy (High NDVI, High NDWI)
            ee.Feature(None, {'NDVI': 0.8, 'NDWI': 0.4, 'NDMI': 0.5, 'MSI': 0.5, 'NDRE': 0.7, 'class': 0}),
            ee.Feature(None, {'NDVI': 0.7, 'NDWI': 0.3, 'NDMI': 0.4, 'MSI': 0.6, 'NDRE': 0.6, 'class': 0}),
            # Water Stress (Low NDWI, High MSI)
            ee.Feature(None, {'NDVI': 0.4, 'NDWI': -0.2, 'NDMI': 0.1, 'MSI': 1.5, 'NDRE': 0.3, 'class': 1}),
            ee.Feature(None, {'NDVI': 0.3, 'NDWI': -0.3, 'NDMI': 0.0, 'MSI': 1.8, 'NDRE': 0.2, 'class': 1}),
            # Heat Stress (High MSI)
            ee.Feature(None, {'NDVI': 0.5, 'NDWI': 0.1, 'NDMI': 0.2, 'MSI': 2.0, 'NDRE': 0.4, 'class': 2}),
            # Disease (Low NDVI, Moderate Water)
            ee.Feature(None, {'NDVI': 0.2, 'NDWI': 0.0, 'NDMI': 0.1, 'MSI': 1.0, 'NDRE': 0.1, 'class': 4}),
        ])

        # Train the classifier
        self.classifier = ee.Classifier.smileRandomForest(10).train(
            features=training_features,
            classProperty='class',
            inputProperties=self.feature_names
        )
        print("Model training command sent to GEE.")

    def validate(self):
        """
        Validates the model accuracy using the training data (Self-Validation).
        In production, this should use a separate testing partition.
        """
        if self.classifier is None:
            self.train()
            
        print("Validating model accuracy...")
        # Classify the training data to check for consistency
        # In a real scenario, use .randomColumn() to split Train/Test
        train_accuracy = self.classifier.confusionMatrix()
        
        print(f"Training Overall Accuracy: {train_accuracy.accuracy().getInfo():.2f}")
        print(f"Kappa Coefficient: {train_accuracy.kappa().getInfo():.2f}")
        print("Confusion Matrix:")
        print(train_accuracy.getInfo())
        return train_accuracy.accuracy().getInfo()

    def classify(self, image):
        """
        Classifies the image using the trained classifier.
        
        Args:
            image (ee.Image): Image containing the spectral indices bands.
            
        Returns:
            ee.Image: Image with an added 'stress_class' band.
        """
        if self.classifier is None:
            self.train()
            # Optional: Validate on first run to ensure health
            # self.validate()
            
        # Classify the image
        classified = image.select(self.feature_names).classify(self.classifier)
        return classified.rename('stress_class')

if __name__ == "__main__":
    from src.data.gee_fetcher import initialize_gee
    try:
        initialize_gee()
        model = CropStressModel()
        model.train()
        model.validate()
        print("Training and Validation successful! (Note: The model lives on GEE servers, not locally)")
    except Exception as e:
        print(f"Error: {e}")
