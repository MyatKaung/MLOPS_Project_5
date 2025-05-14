import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight # Added for sample weighting
from src.logger import get_logger
from src.custom_exception import CustomException

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, processed_data_path="artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info("Model Training Initialization...")

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))
            logger.info("Data loaded for Model")
        except Exception as e:
            logger.error(f"Error while loading data for model: {e}")
            raise CustomException("Failed to load data for model.")
        
    def train_model(self):
        try:
            # Calculate sample weights for class imbalance
            # This assumes self.y_train contains numerically encoded labels (e.g., 0 and 1),
            # which should be the case if data_processing.py's chi2 selection worked correctly.
            sample_weights = compute_sample_weight(class_weight='balanced', y=self.y_train)
            logger.info("Sample weights calculated.")

            # Initialize Gradient Boosting Classifier with parameters from your notebook
            self.model = GradientBoostingClassifier(
                n_estimators=300,       # From notebook
                learning_rate=0.05,     # From notebook
                max_depth=4,            # From notebook
                random_state=42
            )
            
            # Train the model using sample weights
            logger.info("Starting model training with sample weights...")
            self.model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
            logger.info("Model training with sample weights completed.")

            joblib.dump(self.model, os.path.join(self.model_dir, "model.pkl"))
            logger.info("Model trained and saved successfully.")

        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException("Failed to train model.")
        
    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            # Using average='weighted' for overall performance. 
            # Add zero_division=0 to handle cases where a class might not have predictions (though less likely with weighted avg)
            precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("Precision", precision)  # Corrected typo from "Precison"
            mlflow.log_metric("Recall Score", recall)   # Corrected spacing from "Recall Score "
            mlflow.log_metric("F1_score", f1)

            logger.info(f"Accuracy: {accuracy:.4f}; Precision: {precision:.4f}; Recall: {recall:.4f}; F1-Score: {f1:.4f}")

            # Calculate ROC-AUC score only if the target is binary
            unique_y_test = np.unique(self.y_test)
            if len(unique_y_test) == 2:
                try:
                    y_proba = self.model.predict_proba(self.X_test)[:, 1]
                    roc_auc = roc_auc_score(self.y_test, y_proba)
                    mlflow.log_metric("ROC-AUC", roc_auc)
                    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
                except Exception as e:
                    logger.warning(f"Could not compute ROC-AUC score: {e}")
            else:
                logger.warning(f"ROC-AUC score not computed. Target variable in test set is not binary (unique values: {unique_y_test}).")

            logger.info("Model evaluation done.")

        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model.")
        
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__=="__main__":
    # Ensure you have an active MLflow run if you intend to log metrics
    # This can be managed outside or explicitly started here.
    # Example: with mlflow.start_run(run_name="GradientBoostingWeightedTraining"):
    with mlflow.start_run(): # Or use a more descriptive run name
        trainer = ModelTraining()
        trainer.run()