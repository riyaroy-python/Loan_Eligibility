import logging
import matplotlib.pyplot as plt
import os

def save_plot(fig, filename: str):
    """
    Save a matplotlib figure to the plots folder.
    
    Parameters:
        fig (matplotlib.figure.Figure): Figure object to save.
        filename (str): The filename to save the figure as.
    """
    try:
        os.makedirs("plots", exist_ok=True)
        filepath = os.path.join("plots", filename)
        fig.savefig(filepath)
        logging.info(f"Plot saved as {filepath}")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
        raise e

def plot_feature_importance(feature_names, importances, title="Feature Importances"):
    """
    Plot feature importances.
    
    Parameters:
        feature_names (list): List of feature names.
        importances (list or array): Feature importance values.
        title (str): Title of the plot.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_names, importances)
        ax.set_title(title)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Failed to plot feature importance: {e}")
        raise e

if __name__ == '__main__':
    # Example usage of plotting function
    features = ['A', 'B', 'C']
    importances = [0.2, 0.5, 0.3]
    fig = plot_feature_importance(features, importances)
    save_plot(fig, "feature_importance.png")
