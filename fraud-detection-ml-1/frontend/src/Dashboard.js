import React, { useEffect, useState } from "react";
import axios from "axios";
import { Bar, Pie } from "react-chartjs-2";

function Dashboard() {
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    // Fetch transactions periodically
    const interval = setInterval(() => {
      axios
        .get("http://localhost:5000/transactions")
        .then((response) => {
          setTransactions(response.data.transactions);
        })
        .catch((error) => {
          console.error("Error fetching transactions:", error);
        });
    }, 5000); // Fetch every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const getChartData = () => {
    const fraudCount = transactions.filter(
      (t) => t.status === "Fraudulent"
    ).length;
    const nonFraudCount = transactions.length - fraudCount;

    return {
      labels: ["Fraudulent", "Non-Fraudulent"],
      datasets: [
        {
          data: [fraudCount, nonFraudCount],
          backgroundColor: ["#FF6384", "#36A2EB"],
        },
      ],
    };
  };

  return (
    <div className="dashboard">
      <h3>Real-Time Transaction Monitoring</h3>
      <ul>
        {transactions.map((transaction) => (
          <li key={transaction.transaction_id}>
            {transaction.amount} - {transaction.status}
          </li>
        ))}
      </ul>
      <div className="charts">
        <h4>Fraud Detection Summary</h4>
        <Pie data={getChartData()} />
      </div>
    </div>
  );
}

export default Dashboard;
