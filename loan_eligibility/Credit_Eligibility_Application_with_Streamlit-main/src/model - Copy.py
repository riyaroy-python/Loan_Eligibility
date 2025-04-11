import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def train_model(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    Train a simple Logistic Regression model on the dataset.
    
    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame.
        target (str): The name of the target column.
        test_size (float): Proportion of data to use as test set.
        random_state (int): Random state for data splitting.
    
    Returns:
        model, X_test, y_test: Trained model and test data for evaluation.
    """
    try:
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logging.info(f"Model accuracy: {acc:.4f}")
        logging.info("Classification Report:")
        logging.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_filepath = os.path.join("models", "credit_model.joblib")
        joblib.dump(model, model_filepath)
        logging.info(f"Model saved to {model_filepath}")
        
        return model, X_test, y_test
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

if __name__ == '__main__':
    # For testing: load preprocessed data and train the model
    from preprocess import preprocess_data
    from data_loader import load_data
    
    try:
        df_raw = load_data(os.path.join("data", "credit.csv"))
        df_processed = preprocess_data(df_raw)
        
        # Replace 'target' with your actual target column name in the dataset.
        target_col = 'target'
        if target_col not in df_processed.columns:
            logging.error(f"Target column '{target_col}' not found in data.")
        else:
            train_model(df_processed, target=target_col)
    except Exception as e:
        logging.error(f"Model training failed: {e}")
