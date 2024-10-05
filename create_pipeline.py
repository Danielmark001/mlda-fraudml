import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your training data
# Update the path with the actual path to your training dataset
training_data_path = r'uploads\PS_20174392719_1491204439457_log.csv'
df = pd.read_csv(training_data_path)

# Define the features and target
target = 'isFraud'  # Replace with your target column name
features = df.drop(columns=[target], axis=1)
y = df[target]

# Add new columns for balance change calculations to avoid division by zero
features['balance_change_orig'] = (features['newbalanceOrig'] - features['oldbalanceOrg']) / (features['oldbalanceOrg'] + 1e-9)
features['balance_change_dest'] = (features['newbalanceDest'] - features['oldbalanceDest']) / (features['oldbalanceDest'] + 1e-9)

# Remove columns that are not needed for training
columns_to_remove = ['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
features = features.drop(columns=columns_to_remove, errors='ignore')

# Use only the features that were used to train the model
features_to_use = ['type', 'amount', 'balance_change_orig', 'balance_change_dest']
X = features[features_to_use]

# Define which features are numeric and which are categorical
numeric_features = ['amount', 'balance_change_orig', 'balance_change_dest']
categorical_features = ['type']

# Create transformers for numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Combine transformers into a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a complete pipeline that first preprocesses the data
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit the pipeline on the training data
pipeline.fit(X)

# Save the fitted preprocessing pipeline to a file
pipeline_file_path = 'preprocessing_pipeline.pkl'  # The name of the file to save the pipeline
with open(pipeline_file_path, 'wb') as f:
    pickle.dump(pipeline, f)

print(f"Preprocessing pipeline saved successfully to '{pipeline_file_path}'")
