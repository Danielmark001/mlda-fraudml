import React, { useState } from "react";
import axios from "axios";
import "./FileUpload.css"; 

function FileUpload() {
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = () => {
    if (!file) {
      alert("Please select a file to upload.");
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    axios
      .post("http://localhost:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(progress);
        },
      })
      .then((response) => {
        alert("File uploaded successfully.");
        setIsUploading(false);
        setUploadProgress(0); // Reset progress bar
      })
      .catch((error) => {
        alert("File upload failed.");
        setIsUploading(false);
        setUploadProgress(0); // Reset progress bar
      });
  };

  return (
    <div className="file-upload">
      <h3>Upload Transactions CSV</h3>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={isUploading}>
        Upload
      </button>

      {isUploading && (
        <div className="progress-bar-container">
          <div className="progress-bar" style={{ width: `${uploadProgress}%` }}>
            <span className="progress-text">{uploadProgress}%</span>
          </div>
        </div>
      )}

      <a href="/sample-template.csv" download>
        Download Sample CSV Template
      </a>
    </div>
  );
}

export default FileUpload;
