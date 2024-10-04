from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = joblib.load('model/nn_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# In-memory store for incoming transactions (for feedback loop and periodic training)
new_transactions = []

# Endpoint for real-time transaction prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract features from the incoming data
    transaction_data = np.array([
        data['amount'],
        data['balance_change_orig'],
        data['balance_change_dest'],
        # Add other features as required
    ]).reshape(1, -1)
    
    # Scale the input data
    transaction_data_scaled = scaler.transform(transaction_data)
    
    # Make prediction
    prediction = model.predict(transaction_data_scaled)
    is_fraud = bool(prediction[0])  # Convert to boolean (True for fraud, False for non-fraud)
    
    # Store new transaction for feedback loop (in-memory storage)
    new_transactions.append(data)
    
    confidence_score = model.predict_proba(transaction_data_scaled)[0][1] * 100  # Example for binary classification

    # Explanation (You can make this dynamic based on feature importance)
    explanation = "Unusually high transaction amount and balance change detected."

    return jsonify({
        'is_fraud': is_fraud,
        'confidence': confidence_score,
        'explanation': explanation
    })

# Endpoint to periodically update the model with new transactions
@app.route('/update_model', methods=['POST'])
def update_model():
    global model, scaler, new_transactions

    # Convert new transactions into a DataFrame
    if len(new_transactions) == 0:
        return jsonify({'status': 'No new transactions to update model'}), 200

    df_new = pd.DataFrame(new_transactions)
    
    # Feature engineering (similar to what was done during training)
    df_new['balance_change_orig'] = (df_new['newbalanceOrig'] - df_new['oldbalanceOrg']) / (df_new['oldbalanceOrg'] + 1e-9)
    df_new['balance_change_dest'] = (df_new['newbalanceDest'] - df_new['oldbalanceDest']) / (df_new['oldbalanceDest'] + 1e-9)
    
    # Select features and target variable
    X_new = df_new.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    y_new = df_new['isFraud']
    
    # Scale the new data
    X_new_scaled = scaler.transform(X_new)
    
    # Retrain the model using new data
    model.fit(X_new_scaled, y_new)
    
    # Save the updated model
    joblib.dump(model, 'model/model.pkl')
    
    # Clear the in-memory transaction list
    new_transactions = []
    
    return jsonify({'status': 'Model updated successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
