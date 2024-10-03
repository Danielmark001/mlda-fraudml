# Credit Card Fraud Detection
A machine learning-powered application that detects fraudulent transactions. This repository contains both the backend (Flask) and frontend (React) code for processing transactions, allowing users to upload data for batch processing, and monitoring real-time transaction status using a dashboard.

## Features
- File Upload: Allows users to upload a CSV file containing multiple transactions for batch processing.
- Real-Time Monitoring: Displays incoming transactions and their fraud status in real-time.
- Progress Bar: Shows upload progress during file processing.
- Transaction Filtering: Includes options to filter transactions by amount, risk level, and date.
- Summary Statistics: Visualizes key metrics like transaction distribution and fraud detection rates.
  
## Technologies Used
- Frontend: React, Axios, HTML, CSS
- Backend: Flask, Flask-CORS, Pandas, Joblib
- Model: Machine learning model (Decision Trees, MLP Classifier, Ramdom Forest Model, Neural Network, Gradient Boosting) trained for fraud detection using a transaction dataset.
- Additional Libraries: Chart.js or Recharts for graphs and visualizations (frontend).
  
## Getting Started
### Prerequisites
- Node.js and npm installed on your machine.
- Python 3 and pip installed.
- Recommended Python libraries: flask, flask-cors, pandas, joblib, sklearn.
  
## Installation
1. Clone the Repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
2. Backend Setup (Flask)
Navigate to the backend directory and set up the Flask server.
```bash
Copy code
cd backend
pip install -r requirements.txt

3. Frontend Setup (React)
Navigate to the frontend directory and install the necessary packages.

```bash
Copy code
cd ../frontend
npm install

## Running the Application
1. Start the Backend (Flask)
bash
Copy code
cd backend
python app.py
The backend will run on http://localhost:5000.

2. Start the Frontend (React)
In a new terminal window, start the React development server:

bash
Copy code
cd frontend
npm start
The frontend will run on http://localhost:3000.
Usage
Upload a CSV File: On the web interface, use the file upload section to upload a transactions CSV for batch processing. You can download a sample CSV template from the provided link to format your data correctly.
Monitor Transactions: Use the real-time monitoring dashboard to view the transaction feed, apply filters, and see the model's fraud status for each transaction.
Customize and Filter: Adjust filters (e.g., transaction amount, risk level, date range) to navigate through the transaction data efficiently.
Feedback: Use the "Mark as Incorrect" button to provide feedback on the model's predictions to improve future retraining.
File Structure
bash
Copy code
credit-card-fraud-detection/

Sample CSV Template
A sample CSV template (sample-template.csv) is provided in the frontend/public directory. Users can download and format their transaction data accordingly for batch processing.

API Endpoints
POST /upload: Uploads a CSV file for batch processing.
GET /transactions: Retrieves the latest transactions for real-time monitoring.
POST /feedback: Allows users to submit feedback on the model's predictions for future improvements.
