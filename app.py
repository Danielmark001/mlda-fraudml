import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Set TensorFlow log level to suppress excessive logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up the Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'static/plots'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOTS_FOLDER'] = PLOTS_FOLDER

# Load the pre-trained model
model = load_model('nn_model (1).h5')  # Make sure the correct model is used

# In-memory storage for new transactions (for feedback loop or future retraining)
new_transactions = []

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is in the request
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        
        # Check if file is selected and allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Load CSV into pandas dataframe
                df = pd.read_csv(filepath)

                # Validate the necessary column exists
                if 'isFraud' not in df.columns:
                    return "Error: 'isFraud' column not found in the dataset.", 400

                y = df['isFraud']
                
                # Add new columns for balance change calculations to avoid division by zero
                df['balance_change_orig'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + 1e-9)
                df['balance_change_dest'] = (df['newbalanceDest'] - df['oldbalanceDest']) / (df['oldbalanceDest'] + 1e-9)

                # Remove unnecessary columns
                columns_to_remove = ['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
                df = df.drop(columns=columns_to_remove, errors='ignore')

                # Use only the features that match the model's expected input
                features_to_use = ['type', 'amount', 'balance_change_orig', 'balance_change_dest']
                
                # Check if all necessary columns are present
                if not all(col in df.columns for col in features_to_use):
                    return f"Error: The dataset does not contain the necessary columns. Expected {features_to_use}", 400
                
                # Filter to only the necessary columns
                X = df[features_to_use].copy()  # Create a copy to avoid SettingWithCopyWarning

                # Handle categorical column 'type' with controlled one-hot encoding
                transaction_types = ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'CASH_IN', 'DEBIT']
                for transaction_type in transaction_types:
                    X.loc[:, f'type_{transaction_type}'] = (X['type'] == transaction_type).astype(int)

                # Drop the original 'type' column since it's now one-hot encoded
                X = X.drop(columns=['type'])

                # Scale numeric features using StandardScaler
                numeric_features = ['amount', 'balance_change_orig', 'balance_change_dest']
                scaler = StandardScaler()
                X_numeric = scaler.fit_transform(X[numeric_features])
                X_numeric = pd.DataFrame(X_numeric, columns=numeric_features)

                # Concatenate the processed numeric and categorical features
                X_categorical = X.drop(columns=numeric_features)
                X_preprocessed = pd.concat([X_numeric, X_categorical], axis=1)

                # Ensure the number of features matches what the model expects
                if X_preprocessed.shape[1] != model.input_shape[-1]:
                    return f"Error: The number of features ({X_preprocessed.shape[1]}) does not match the expected input ({model.input_shape[-1]}).", 400

                # Make predictions
                y_pred_prob = model.predict(X_preprocessed)
                y_pred = (y_pred_prob > 0.3).astype(int).flatten()  # Using a threshold of 0.3

                # Calculate metrics
                f1 = f1_score(y, y_pred, average='weighted')
                precision = precision_score(y, y_pred, average='weighted')
                recall = recall_score(y, y_pred, average='weighted')
                accuracy = accuracy_score(y, y_pred)

                # Calculate ROC curve and AUC
                fpr, tpr, _ = roc_curve(y, y_pred_prob)
                roc_auc = auc(fpr, tpr)

                # Plot ROC curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                
                # Save the plot
                plot_path = os.path.join(app.config['PLOTS_FOLDER'], 'roc_curve.png')
                plt.savefig(plot_path)
                plt.close()

                # Prepare fraud predictions for display
                fraud_predictions = []
                for idx, prob in enumerate(y_pred_prob):
                    is_fraud = "Fraud" if y_pred[idx] == 1 else "Not Fraud"
                    fraud_percentage = prob[0] * 100
                    fraud_predictions.append({
                        "transaction_index": idx,
                        "is_fraud": is_fraud,
                        "fraud_percentage": f"{fraud_percentage:.2f}%"
                    })

                # Return the metrics, plot, and predictions
                return render_template('index.html', 
                                       f1=f1, 
                                       precision=precision, 
                                       recall=recall, 
                                       accuracy=accuracy, 
                                       roc_auc=roc_auc, 
                                       plot_url=url_for('static', filename='plots/roc_curve.png'), 
                                       fraud_predictions=fraud_predictions)

            except Exception as e:
                return f"An error occurred while processing the file: {str(e)}", 500

        return "Invalid file format. Please upload a CSV file.", 400

    # GET request - just render the form
    return render_template('index.html')

if __name__ == "__main__":
    # Create necessary folders if they don't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)
    
    # Start the Flask application
    app.run(debug=True)
