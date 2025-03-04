import { useState } from 'react'
//import reactLogo from './assets/react.svg'
//import viteLogo from '/vite.svg'
import axios from "axios";
import './App.css'

function App() {
  const [text, setText] = useState("");
  // const [result, setResult] = useState("");
  // const [model, setModel] = useState("ml");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Send a POST request to the Flask backend
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        text: text,
      });

      // Log the API response
      console.log("API Response:", response.data);

      // Update the prediction state
      setPrediction(response.data.prediction);
      console.log("Prediction State:", prediction);
    } catch (error) {
      console.error("Error making prediction:", error);
      setError("Failed to make prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Fake News Detector</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter news text..."
          rows={5}
          cols={50}
          required
        />
        <br />
        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {prediction !== null && (
        <div>
          <h2>Prediction:</h2>
          <p>{prediction === 1 ? "Fake News" : "Real News"}</p>
        </div>
      )}
    </div>
  );
};

export default App;

//   const handleSubmit = async () => {
//     const response = await fetch("http://127.0.0.1:5000/predict", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ text, model }),
//     });
//     const data = await response.json();
//     setResult(data.prediction);
//   };

//   return (
//     <div style={{ padding: "20px" }}>
//       <h1>Fake News Detector</h1>
//       <textarea
//         placeholder="Paste news content..."
//         value={text}
//         onChange={(e) => setText(e.target.value)}
//         rows={5}
//         cols={50}
//       />
//       <br />
//       <select value={model} onChange={(e) => setModel(e.target.value)}>
//         <option value="ml">Traditional ML</option>
//         <option value="bert">Deep Learning (BERT)</option>
//       </select>
//       <br />
//       <button onClick={handleSubmit}>Detect</button>
//       <h3>Result: {result}</h3>
//     </div>
//   );
// }

// export default App;