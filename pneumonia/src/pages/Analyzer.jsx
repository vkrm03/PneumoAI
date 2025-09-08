import React, { useState } from "react";

function Analyzer() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null); // reset old result
    }
  };

  const handleAnalyze = () => {
    if (!image) {
      alert("Please choose an X-ray image first.");
      return;
    }

    // Mock AI analysis result
    const randomConfidence = (90 + Math.random() * 10).toFixed(2) + "%";
    const mockResult = {
      prediction: Math.random() > 0.5 ? "Pneumonia Detected" : "Normal",
      confidence: randomConfidence,
    };

    setResult(mockResult);
  };

  return (
    <div className="p-8 text-center">
      <h2 className="text-2xl font-bold mb-4 text-white">Upload Chest X-ray</h2>

      {/* Hidden file input */}
      <input
        id="fileInput"
        type="file"
        accept="image/*"
        onChange={handleUpload}
        className="hidden"
      />

      {/* Choose Image button */}
      <label
        htmlFor="fileInput"
        className="px-4 py-2 bg-gray-600 text-white rounded cursor-pointer hover:bg-gray-700 mr-4"
      >
        Choose Image
      </label>

      {/* Upload & Analyze button */}
      <button
        onClick={handleAnalyze}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        Upload & Analyze
      </button>

      {preview && result && (
        <div className="mt-10 bg-white p-6 rounded-lg shadow-lg max-w-5xl mx-auto">
          {/* Analysis Results */}
          <div className="text-left mb-6">
            <p><strong>Prediction:</strong> {result.prediction}</p>
            <p><strong>Confidence:</strong> {result.confidence}</p>
          </div>

          {/* Uploaded and Digital Images */}
          <div className="flex justify-center gap-8">
            <div>
              <h3 className="font-semibold mb-2">Uploaded Image</h3>
              <img
                src={preview}
                alt="Uploaded X-ray"
                className="w-64 h-64 object-contain border"
              />
            </div>
            <div>
              <h3 className="font-semibold mb-2">Digital Image</h3>
              <img
                src={preview}
                alt="Digital X-ray"
                className="w-64 h-64 object-contain border filter grayscale contrast-200"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Analyzer;
