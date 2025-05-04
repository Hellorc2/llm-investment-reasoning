import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score

# Read the batch 1 predictions file
df = pd.read_csv('vanilla_openai_test_data_reduced_with_definition_restricted.csv')

prediction_numeric = df['prediction'].apply(lambda x: 1 if x == 'Yes' else 0)

# Calculate metrics using numeric predictions
precision = precision_score(df['success'], prediction_numeric)
accuracy = accuracy_score(df['success'], prediction_numeric)
recall = recall_score(df['success'], prediction_numeric)

print(f"Precision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")

# Print additional information
print(f"\nTotal 2samples analyzed: {len(df)}")

# Create a confusion matrix
confusion_matrix = pd.crosstab(df['success'], prediction_numeric, rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix:")
print(confusion_matrix)