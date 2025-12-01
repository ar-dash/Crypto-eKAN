import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Basic Metrics Dictionary
def get_metrics(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

# === Scatter Plot: Actual vs Predicted
def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted Values"):
    errors = np.abs(y_true - y_pred)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(y_true, y_pred, c=errors, cmap='viridis', alpha=0.6, edgecolors='k', s=60)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit", linewidth=2)
    cbar = plt.colorbar(sc)
    cbar.set_label("Prediction Error")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# === Residual Plot
def plot_residuals(y_pred, y_true):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(y_pred, residuals, c=np.abs(residuals), cmap='coolwarm', edgecolors='k', alpha=0.6, s=60)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
    cbar = plt.colorbar(sc)
    cbar.set_label("Absolute Residuals")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# === Histogram of Residuals
def plot_residual_histogram(residuals):
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=25, kde=True, color='blue', alpha=0.6, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# === Learning Curve Visualization
def plot_learning_curve(train_losses, val_losses, title="Learning Curve"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', markersize=6)
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss', markersize=6)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
