# eda_module.py
# Perform Statistics, Train models and generate Visualisation 

# Required libraries for eda module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class DataAnalyzer:
    """Class to perform exploratory data analysis, handle class imbalance, and train ML models."""
    
    def __init__(self, data, X_train, y_train, X_test, y_test):
        """Initialize with dataset and train/test splits."""
        self._data = data
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._output_dir = 'eda_plots'
        self._ml_output_dir = 'ml_plots'
        os.makedirs(self._output_dir, exist_ok=True)
        os.makedirs(self._ml_output_dir, exist_ok=True)
        self._models = {
            'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),  
            'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)  
        }
    
    def compute_statistics(self):
        """Compute descriptive statistics for numerical features."""
        numerical_cols = ['Age', 'Average Glucose Level', 'BMI', 'Sleep Hours', 'Stroke Risk Score']
        stats = {}
        
        for col in numerical_cols:
            if col in self._data.columns:
                stats[col] = {
                    'Mean': self._data[col].mean(),
                    'Median': self._data[col].median(),
                    'Std': self._data[col].std(),
                    'Variance': self._data[col].var(),
                    'Min': self._data[col].min(),
                    'Max': self._data[col].max(),
                    'Skewness': self._data[col].skew(),
                    'Kurtosis': self._data[col].kurtosis()
                }
        
        # Convert to DataFrame for display
        stats_df = pd.DataFrame(stats).T
        print("Descriptive Statistics:\n", stats_df)
        stats_df.to_csv(os.path.join(self._output_dir, 'descriptive_statistics.csv'))
        return stats_df

    def compute_derived_features(self):
        """Compute derived features like interaction terms."""
        if 'Age' in self._data.columns and 'Average Glucose Level' in self._data.columns:
            self._data['Age_Glucose_Interaction'] = self._data['Age'] * self._data['Average Glucose Level']
            print("Computed interaction feature: Age_Glucose_Interaction")
        return self._data
    
    def visualize_data(self):
        """Generate visualizations for EDA."""
        
        # Histogram for numerical features
        numerical_cols = ['Age', 'Average Glucose Level', 'BMI', 'Sleep Hours', 'Stroke Risk Score']
        for col in numerical_cols:
            if col in self._data.columns:
                plt.figure(figsize=(8, 6))
                sns.histplot(self._data[col], bins=30, kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.savefig(os.path.join(self._output_dir, f'{col}_histogram.png'))
                plt.close()
        
        # Pie chart for categorical features
        categorical_cols = ['Gender', 'Smoking Status', 'Physical Activity', 'Income Level']
        for col in categorical_cols:
            if col in self._data.columns:
                plt.figure(figsize=(8, 6))
                self._data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
                plt.title(f'Distribution of {col}')
                plt.ylabel('')
                plt.savefig(os.path.join(self._output_dir, f'{col}_pie.png'))
                plt.close()
    
        # Scatter plot for feature dependencies
        if 'Age' in self._data.columns and 'Average Glucose Level' in self._data.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='Age', y='Average Glucose Level', hue='Stroke Occurrence', data=self._data)
            plt.title('Age vs. Glucose Level by Stroke Occurrence')
            plt.savefig(os.path.join(self._output_dir, 'age_vs_glucose_scatter.png'))
            plt.close()
    
        # Grouped bar plot for categorical features vs. Stroke Occurrence
        if 'Smoking Status' in self._data.columns and 'Stroke Occurrence' in self._data.columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(x='Smoking Status', hue='Stroke Occurrence', data=self._data)
            plt.title('Smoking Status by Stroke Occurrence')
            plt.savefig(os.path.join(self._output_dir, 'smoking_vs_stroke_bar.png'))
            plt.close()
                    
        # Box plot for numerical vs. Stroke Occurrence
        if 'Stroke Occurrence' in self._data.columns:
            for col in numerical_cols:
                if col in self._data.columns:
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(x='Stroke Occurrence', y=col, data=self._data)
                    plt.title(f'{col} by Stroke Occurrence')
                    plt.savefig(os.path.join(self._output_dir, f'{col}_vs_stroke_boxplot.png'))
                    plt.close()
        
        # Correlation heatmap
        numerical_data = self._data.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(self._output_dir, 'correlation_heatmap.png'))
        plt.close()
        
        print(f"EDA visualizations saved in {self._output_dir}")
    
    def check_class_balance(self):
        """Check class balance for target variables."""
        target_columns = ['Chronic Stress', 'Physical Activity', 'Income Level', 'Stroke Occurrence']
        for col in target_columns:
            if col in self._data.columns:
                plt.figure(figsize=(8, 6))
                sns.countplot(x=col, data=self._data)
                plt.title(f'Class Distribution for {col}')
                plt.savefig(os.path.join(self._output_dir, f'{col}_class_balance.png'))
                plt.close()
                print(f"Class distribution for {col}:\n", self._data[col].value_counts())
    
    def balance_classes(self, target_column):
        """Apply SMOTE to balance classes for a specific target."""
        smote = SMOTE(random_state=42, k_neighbors=3)  # Reduced k_neighbors for speed
        try:
            X_balanced, y_balanced = smote.fit_resample(self._X_train, self._y_train[target_column])
            print(f"SMOTE applied for {target_column}. New class distribution:\n", pd.Series(y_balanced).value_counts())
            return X_balanced, y_balanced
        except ValueError as e:
            print(f"Error applying SMOTE for {target_column}: {str(e)}")
            return self._X_train, self._y_train[target_column]
    
    def train_and_evaluate(self, target):
        """Train and evaluate models for a specific target variable."""
        results = {}
        # Apply SMOTE for imbalanced targets (e.g., Stroke Occurrence, Chronic Stress)
        if target in ['Stroke Occurrence', 'Chronic Stress'] and len(np.unique(self._y_train[target])) > 1:
            X_train_bal, y_train_bal = self.balance_classes(target)
        else:
            X_train_bal, y_train_bal = self._X_train, self._y_train[target]
        
        for name, model in self._models.items():
            try:
                # Train model
                model.fit(X_train_bal, y_train_bal)
                # Predict
                y_pred = model.predict(self._X_test)
                # Compute metrics
                results[name] = {
                    'accuracy': accuracy_score(self._y_test[target], y_pred),
                    'precision': precision_score(self._y_test[target], y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(self._y_test[target], y_pred, average='weighted', zero_division=0),
                    'confusion_matrix': confusion_matrix(self._y_test[target], y_pred)
                }
            except Exception as e:
                print(f"Error training {name} for {target}: {str(e)}")
        return results
    
    def train_and_evaluate_all(self):
        """Train and evaluate models for all target variables."""
        targets = ['Chronic Stress', 'Physical Activity', 'Income Level', 'Stroke Occurrence']
        all_results = {}
        for target in targets:
            if target in self._y_train.columns:
                print(f"Training models for {target}...")
                all_results[target] = self.train_and_evaluate(target)
        
        # Plot confusion matrices and model comparisons after training
        self._plot_results(all_results)
        return all_results
    
    def _plot_results(self, results):
        """Plot confusion matrices and model comparisons."""
        # Plot confusion matrices
        for target in results:
            for name in results[target]:
                plt.figure(figsize=(8, 6))
                sns.heatmap(results[target][name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix: {name} ({target})')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.savefig(os.path.join(self._ml_output_dir, f'{target}_{name}_confusion_matrix.png'))
                plt.close()
                
        # Plot model comparison
        metrics = ['accuracy', 'precision', 'recall']
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            model_names = list(self._models.keys())
            bar_width = 0.2
            for i, target in enumerate(results):
                scores = [results[target][model][metric] for model in model_names]
                x = np.arange(len(model_names)) + i * bar_width
                plt.bar(x, scores, bar_width, label=target)
            
            plt.title(f'Model Comparison: {metric.capitalize()}')
            plt.xticks(np.arange(len(model_names)) + bar_width * (len(results) - 1) / 2, model_names)
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self._ml_output_dir, f'model_comparison_{metric}.png'))
            plt.close()

    def analyze_model_performance(self, results):
        for target in results:
            print(f"\nAnalysis for {target}:")
            for name in results[target]:
                print(f"{name}: Accuracy={results[target][name]['accuracy']:.2f}, "
                      f"Precision={results[target][name]['precision']:.2f}, "
                      f"Recall={results[target][name]['recall']:.2f}")
                if results[target][name]['accuracy'] > 0.85:
                    print(f"{name} performs well due to robust handling of {target} patterns.")
    
    def compute_statistical_features(self):
        """Compute statistical features for machine learning."""
        feature_importance = {}
        if 'Stroke Occurrence' in self._data.columns:
            numerical_data = self._data.select_dtypes(include=[np.number])
            correlations = numerical_data.corr()['Stroke Occurrence'].sort_values(ascending=False)
            feature_importance['Stroke Occurrence'] = correlations
            print("Feature correlations with Stroke Occurrence:\n", correlations)

            # Save feature importance to CSV
            pd.Series(correlations).to_csv(
                os.path.join(self._output_dir, 'feature_importance.csv')
            )
        return feature_importance
