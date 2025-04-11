import pandas as pd
import logging
import os

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "data_loader.log")),
        logging.StreamHandler()
    ]
)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the credit dataset from a CSV file.
    
    Parameters:
        filepath (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded data.
    
    Raises:
        FileNotFoundError: If the CSV file is not found.
        Exception: For any other errors that occur during data loading.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {filepath}")
        raise fnf_error
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        raise e

def initial_analysis(df: pd.DataFrame):
    """
    Perform initial exploration of the DataFrame: head, summary statistics,
    and missing values check.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    logging.info("First 5 rows of the data:")
    logging.info(f"\n{df.head()}")
    
    logging.info("Summary statistics:")
    logging.info(f"\n{df.describe(include='all')}")
    
    missing_values = df.isnull().sum()
    logging.info("Missing values in each column:")
    logging.info(f"\n{missing_values}")
    
if __name__ == '__main__':
    # Define the CSV file path (adjust path if your file is in a different location)
    filepath = os.path.join("data", "credit.csv")
    
    try:
        data = load_data(filepath)
        initial_analysis(data)
    except Exception as error:
        logging.error(f"Failed to complete initial analysis: {error}")
