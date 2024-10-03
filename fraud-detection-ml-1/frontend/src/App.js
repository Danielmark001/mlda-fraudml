import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [formData, setFormData] = useState({
        amount: '',
        balance_change_orig: '',
        balance_change_dest: ''
    });
    const [prediction, setPrediction] = useState(null);
    const [confidence, setConfidence] = useState(null);
    const [riskLevel, setRiskLevel] = useState(null);
    const [explanation, setExplanation] = useState('');

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://localhost:5000/predict', formData);

            // Extract response data
            setPrediction(response.data.is_fraud);
            setConfidence(response.data.confidence);  // Assuming backend provides confidence score
            setExplanation(response.data.explanation); // Assuming backend provides an explanation

            // Determine risk level based on confidence score
            if (response.data.confidence >= 80) {
                setRiskLevel('High');
            } else if (response.data.confidence >= 50) {
                setRiskLevel('Medium');
            } else {
                setRiskLevel('Low');
            }
        } catch (error) {
            console.error('Error while fetching the prediction:', error);
        }
    };

    return (
        <div className="App">
            <h1>Credit Card Fraud Detection</h1>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label>Transaction Amount:</label>
                    <input
                        type="number"
                        name="amount"
                        value={formData.amount}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Balance Change (Origin):</label>
                    <input
                        type="number"
                        name="balance_change_orig"
                        value={formData.balance_change_orig}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Balance Change (Destination):</label>
                    <input
                        type="number"
                        name="balance_change_dest"
                        value={formData.balance_change_dest}
                        onChange={handleChange}
                        required
                    />
                </div>
                <button type="submit">Check for Fraud</button>
            </form>

            {/* Display the prediction result */}
            {prediction !== null && (
                <div className="result">
                    <h2>Prediction Result</h2>
                    <p className={`result-text ${prediction ? 'fraud' : 'non-fraud'}`}>
                        {prediction ? 'Fraudulent Transaction Detected' : 'Transaction is Legitimate'}
                    </p>

                    {/* Display confidence score */}
                    {confidence !== null && (
                        <p className="confidence">
                            Confidence: {confidence}%
                        </p>
                    )}

                    {/* Display risk level */}
                    {riskLevel && (
                        <p className={`risk-level ${riskLevel.toLowerCase()}`}>
                            Risk Level: {riskLevel}
                        </p>
                    )}

                    {/* Display explanation of the result */}
                    {explanation && (
                        <div className="explanation">
                            <h3>Explanation</h3>
                            <p>{explanation}</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default App;
