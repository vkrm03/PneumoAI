import React, { useState } from "react";

function Analyzer() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith("image/")) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    } else {
      alert("Only image files are allowed (JPG, PNG, etc.)");
    }
  };

  const handleAnalyze = () => {
    if (!image) {
      alert("Please upload an X-ray image first.");
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
      <h2 className="text-2xl font-bold mb-6 text-white">Upload Chest X-ray</h2>

      {/* Drag & Drop / Click Upload Zone */}
      <div
        className="w-full max-w-lg mx-auto p-10 border-4 border-dashed rounded-lg cursor-pointer bg-gray-100 hover:bg-gray-200 transition"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const file = e.dataTransfer.files[0];
          handleFileSelect(file);
        }}
        onClick={() => document.getElementById("image-upload").click()}
      >
        {preview ? (
          <img
            src={preview}
            alt="Uploaded Preview"
            className="mx-auto w-64 h-64 object-contain"
          />
        ) : (
          <p className="text-gray-600">
            Drag & Drop X-ray here or <span className="font-semibold">Click to Browse</span>
          </p>
        )}
      </div>

      {/* Hidden file input for click fallback */}
      <input
        type="file"
        id="image-upload"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => handleFileSelect(e.target.files[0])}
      />

      {/* Upload & Analyze button */}
      <button
        onClick={handleAnalyze}
        className="mt-6 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
      >
        Upload & Analyze
      </button>

      {preview && result && (
        <div className="mt-10 bg-white p-6 rounded-lg shadow-lg max-w-5xl mx-auto">
          {/* Analysis Results */}
          <div className="text-left mb-6">
            <p>
              <strong>Prediction:</strong> {result.prediction}
            </p>
            <p>
              <strong>Confidence:</strong> {result.confidence}
            </p>
          </div>

          {/* Uploaded and Finding Images */}
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
              <h3 className="font-semibold mb-2">Finding Image</h3>
              <img
                src={preview}
                alt="Finding X-ray"
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
