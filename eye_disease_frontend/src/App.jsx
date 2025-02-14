import React, { useState } from "react";
import axios from "axios";
import { Upload, AlertCircle, Loader2 } from "lucide-react";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        setError("File size should be less than 5MB");
        return;
      }
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPredictions(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPredictions(response.data);
    } catch (err) {
      console.error("Error:", err);
      setError(err.response?.data?.error || "An error occurred during prediction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6 min-h-screen flex flex-col justify-center">
      {/* Title */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-800">Eye Disease Predictor</h1>
        <p className="text-gray-600 mt-2">Upload an eye image to analyze possible diseases</p>
      </div>

      {/* Upload Section */}
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center bg-gray-50">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
          id="image-upload"
        />
        <label
          htmlFor="image-upload"
          className="cursor-pointer flex flex-col items-center"
        >
          <Upload className="w-12 h-12 text-gray-500 mb-4" />
          <span className="text-gray-700 font-medium">Click to upload or drag & drop</span>
          <span className="text-sm text-gray-500 mt-1">PNG, JPG, JPEG (max 5MB)</span>
        </label>
      </div>

      {/* Image Preview */}
      {preview && (
        <div className="mt-4 flex flex-col items-center">
          <h2 className="text-lg font-semibold text-gray-800 mb-2">Image Preview</h2>
          <img
            src={preview}
            alt="Preview"
            className="max-w-sm rounded-lg shadow-lg border border-gray-300"
          />
        </div>
      )}

      {/* Analyze Button */}
      <div className="text-center mt-4">
        <button
          onClick={handleSubmit}
          disabled={!selectedFile || loading}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <Loader2 className="animate-spin mr-2" />
              Processing...
            </span>
          ) : (
            "Analyze Image"
          )}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 text-red-700 flex items-center mt-4">
          <AlertCircle className="h-5 w-5 mr-2" />
          {error}
        </div>
      )}

      {/* Results Display */}
      {predictions && (
        <div className="mt-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">Predictions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(predictions).map(([model, result]) => (
              <div
                key={model}
                className="bg-white p-4 rounded-lg shadow border border-gray-200"
              >
                <h3 className="font-semibold text-lg capitalize text-gray-700">
                  {model} Model
                </h3>
                <div className="mt-2 text-gray-600">
                  <p>
                    <span className="font-medium">Predicted Class:</span>{" "}
                    {result.predicted_class}
                  </p>
                  <p>
                    <span className="font-medium">Confidence:</span>{" "}
                    {(result.confidence * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
