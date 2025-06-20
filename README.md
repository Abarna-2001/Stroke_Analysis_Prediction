# STROKE ANALYSIS APPLICATION
This notebook is the main entry point for the Stroke Analysis Application project. It launches a user-friendly Tkinter GUI that lets clinicians explore a stroke dataset.

# Overview
The system is structured into three modular components:
- **load_dataset_module**: Handles data loading, preprocessing (imputation, encoding, scaling), and splitting into training and testing sets.
- **eda_module**: Conducts EDA (descriptive statistics, visualizations) and trains classifiers (Logistic Regression, Random Forest, KNN) with SMOTE for class imbalance.
- **ui_module**: Provides a tkinter-based GUI for user interaction, displaying statistics, model metrics, and plots.

# Requirements
To run the stroke analysis application, ensure the following software and libraries are installed:

## *Software*
- **Python**: Version 3.11 or higher  
- **Operating System**: Windows, macOS, or Linux  
- **Text Editor/IDE**: Jupyter Notebook, VS Code, or similar

## *Python Libraries*
- `pandas` – For data manipulation and DataFrame operations  
- `numpy` – For numerical computations  
- `scikit-learn` – For preprocessing, model training, and evaluation  
- `matplotlib` – For plotting visualizations  
- `seaborn` – For enhanced statistical visualizations  
- `imbalanced-learn` – For SMOTE (handling class imbalance)  
- `pillow` – For image processing in the GUI  
- `tkinter` – Python’s standard library for GUI development
