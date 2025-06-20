# load_dataset_module.py
# Module to load, preprocess, and split the stroke dataset for analysis and modeling

# Required libraries for Load dataset module
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

class DatasetLoader:
    """Handles loading, preprocessing, and splitting of the stroke dataset."""

    def __init__(self, file_path):
        """
        Initialize the loader with the dataset file path.
        Initializes empty data attributes and a scaler for numerical features.
        """
        self._file_path = file_path
        self._data = None
        self._train_data = None
        self._test_data = None
        self._train_labels = None
        self._test_labels = None
        self._scaler = StandardScaler()  # For scaling numerical columns

    def load_data(self):
        """Load dataset CSV into a pandas DataFrame."""
        try:
            self._data = pd.read_csv(self._file_path)
            print(f"Dataset loaded: {self._data.shape[0]} rows, {self._data.shape[1]} columns")
            return self._data
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file '{self._file_path}' not found.")
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")

    def preprocess_data(self):
        """
        Preprocess dataset by imputing missing values, encoding categorical features,
        and scaling numerical columns.
        """
        if self._data is None:
            raise ValueError("No data loaded. Please call load_data() first.")

        # Columns expected to have numerical data to impute and scale
        numerical_cols = ['Age', 'Average Glucose Level', 'BMI', 'Sleep Hours', 'Stroke Risk Score']

        # Impute missing numerical values with median to reduce outlier impact
        for col in numerical_cols:
            if col in self._data.columns:
                self._data[col] = self._data[col].fillna(self._data[col].median())

        # Categorical columns to impute and encode
        categorical_cols = ['Gender', 'Ever Married', 'Work Type', 'Residence Type', 'Smoking Status',
                            'Physical Activity', 'Dietary Habits', 'Alcohol Consumption',
                            'Education Level', 'Income Level', 'Region']

        # Impute missing categorical data with the most frequent value (mode)
        for col in categorical_cols:
            if col in self._data.columns:
                self._data[col] = self._data[col].fillna(self._data[col].mode()[0])

        # Encode categorical columns using LabelEncoder to convert categories into numeric labels
        self._label_encoders = {}
        for col in categorical_cols:
            if col in self._data.columns:
                le = LabelEncoder()
                self._data[col] = le.fit_transform(self._data[col])
                self._label_encoders[col] = le

        # Scale numerical columns to zero mean and unit variance
        existing_numerical_cols = [col for col in numerical_cols if col in self._data.columns]
        if existing_numerical_cols:
            self._data[existing_numerical_cols] = self._scaler.fit_transform(self._data[existing_numerical_cols])

        print("Preprocessing complete: missing values handled, categorical encoded, numerical scaled.")
        return self._data

    def check_class_balance(self, target_column):
        """
        Display the class distribution of the specified target column.
        Useful for detecting class imbalance.
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        distribution = self._data[target_column].value_counts(normalize=True)
        print(f"Class distribution for '{target_column}':\n{distribution}")

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        Separates features and multiple target variables.
        """
        if self._data is None:
            raise ValueError("Data not loaded or preprocessed. Call load_data() and preprocess_data() first.")

        # Define target variables and exclude identifier columns from features
        target_columns = ['Chronic Stress', 'Physical Activity', 'Income Level', 'Stroke Occurrence']
        feature_columns = [col for col in self._data.columns if col not in target_columns + ['ID']]

        X = self._data[feature_columns]
        y = self._data[target_columns]

        # Perform train-test split
        self._train_data, self._test_data, self._train_labels, self._test_labels = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Split data: {len(self._train_data)} training samples, {len(self._test_data)} test samples.")
        return self._train_data, self._test_data, self._train_labels, self._test_labels

    def get_data(self):
        """Return the full dataset along with training and testing splits."""
        return self._data, self._train_data, self._test_data, self._train_labels, self._test_labels
