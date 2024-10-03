import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css"; 
import App from "./app";

const container = document.getElementById("root");
const root = createRoot(container);

// Render the root component into the DOM
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
