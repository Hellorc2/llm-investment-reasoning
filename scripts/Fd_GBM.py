import pandas as pd
import os
from typing import List
import xgboost as xgb
import matplotlib.pyplot as plt
from dtreeviz import dtreeviz

def get_updated_rows(base_csv: str = 'founder_data.csv', 
                    scores_csv: str = 'founder_scores_3_25_reasearch.csv',
                    columns: List[str] = None) -> pd.DataFrame:
    """
    Takes rows from founder_data.csv and finds corresponding rows in founder_scores CSV,
    then returns those rows with specified columns and saves to founder_data_ml.csv.
    
    Args:
        base_csv: Path to the base founder data CSV file (default: founder_data.csv)
        scores_csv: Path to the scores CSV file (default: founder_scores_3_25_reasearch.csv) 
        columns: List of column names to extract from scores CSV
        
    Returns:
        DataFrame containing the matched rows with requested columns
        Or error message if files not found or columns don't exist
    """
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Read both CSV files
    try:
        founder_df = pd.read_csv(os.path.join(parent_dir, base_csv))
        scores_df = pd.read_csv(os.path.join(parent_dir, scores_csv))
    except FileNotFoundError as e:
        print(f"Error: CSV file not found - {str(e)}")
        return None
    except Exception as e:
        print(f"Error reading CSV files: {str(e)}")
        return None
        
    # If no columns specified, use all columns from scores CSV
    if columns is None:
        columns = scores_df.columns.tolist()
    
    # Check if requested columns exist
    invalid_columns = [col for col in columns if col not in scores_df.columns]
    if invalid_columns:
        print(f"Error: The following columns do not exist in {scores_csv}: {', '.join(invalid_columns)}")
        return None
    
    # Add success column if it exists in founder_df
    if 'success' in founder_df.columns:
        scores_df['success'] = founder_df['success']
    
    # Save the resulting dataset to founder_data_ml.csv in parent directory
    output_path = os.path.join(parent_dir, 'founder_data_ml.csv')
    scores_df[columns + ['success']].to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")
    
    return scores_df[columns + ['success']]

def visualize_tree(model, feature_names=None):
    """
    Visualizes the model's feature importance using matplotlib
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (optional)
    """
    try:
        # Get feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance)), importance['importance'])
        plt.yticks(range(len(importance)), importance['feature'])
        plt.xlabel('Feature Importance')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.show()
        
        # Print detailed feature importance
        print("\nDetailed Feature Importance:")
        print(importance.sort_values('importance', ascending=False).head(10))
        
    except Exception as e:
        print(f"Error visualizing tree: {str(e)}")

def print_tree_structure(model, feature_names=None, X_train=None, y_train=None):
    """
    Creates a visual plot of the XGBoost trees
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (optional)
        X_train: Training features
        y_train: Training labels
    """
    try:
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create a bar plot of top 10 features
        plt.figure(figsize=(12, 6))
        plt.barh(range(10), feature_importance['importance'].head(10))
        plt.yticks(range(10), feature_importance['feature'].head(10))
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.show()
        
        # Print detailed feature importance
        print("\nTree Structure Summary:")
        print("----------------------")
        print("Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"\n{row['feature']} (Importance: {row['importance']:.3f})")
            
        # Print the first tree's structure in text format
        print("\nFirst Tree Structure:")
        print("----------------------")
        tree = model.get_booster().trees_to_dataframe()
        print(tree.head(10))  # Show first 10 nodes of the tree
            
    except Exception as e:
        print(f"Error plotting tree structure: {str(e)}")

def train_decision_tree(data_csv: str = 'founder_data_ml.csv'):
    """
    Reads founder data and trains an XGBoost model for prediction
    
    Args:
        data_csv: Path to the processed data CSV file (default: founder_data_ml.csv)
        
    Returns:
        Trained XGBoost model and evaluation metrics
    """
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Read the data
        data = pd.read_csv(os.path.join(parent_dir, data_csv))
        
        # Separate features and target
        X = data.drop('success', axis=1)  
        y = data['success']
        
        # Handle categorical columns
        from sklearn.preprocessing import LabelEncoder
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column].astype(str))
        
        # Fill NaN values with appropriate defaults
        # For numeric columns, use median
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        
        # For categorical columns, use mode
        categorical_columns = X.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            mode_value = X[column].mode()
            if not mode_value.empty:
                X[column] = X[column].fillna(mode_value.iloc[0])
            else:
                # If all values are NaN, fill with a default value
                X[column] = X[column].fillna('Unknown')
        
        # Split data into train and test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            max_depth=5,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Print the number of trees
        print(f"\nNumber of Trees: {model.n_estimators}")
        
        # Make predictions and evaluate
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = model.predict(X_test)
        
        # Print metrics
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Visualize feature importance
        print("\nVisualizing feature importance...")
        visualize_tree(model, feature_names=X.columns.tolist())
        
        # Print tree structure
        print("\nPrinting tree structure...")
        print_tree_structure(model, feature_names=X.columns.tolist(), X_train=X_train, y_train=y_train)
        
        return model
        
    except FileNotFoundError:
        print(f"Error: {data_csv} file not found in the parent directory")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
