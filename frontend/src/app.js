import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  // State to store form input and prediction result
  const [formData, setFormData] = useState({
    amount: "",
    balance_change_orig: "",
    balance_change_dest: "",
  });
  const [prediction, setPrediction] = useState(null);

  // Handle form input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {

      const response = await axios.post(
        "http://localhost:5000/predict",
        formData
      );
      setPrediction(response.data);
    } catch (error) {
      console.error("Error while fetching the prediction:", error);
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
      {prediction && (
        <div className="result">
          <h2>Prediction Result</h2>
          <p>
            {prediction.is_fraud
              ? "Fraudulent Transaction Detected"
              : "Transaction is Legitimate"}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
