import { useState } from 'react'
//import reactLogo from './assets/react.svg'
//import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState("");
  const [model, setModel] = useState("ml");

  const detectFakeNews = async () => {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
    });
    const data = await response.json();
    setResult(data.prediction);
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Fake News Detector</h1>
      <textarea
        placeholder="Paste news content..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={5}
        cols={50}
      />
      <br />
      <select value={model} onChange={(e) => setModel(e.target.value)}>
        <option value="ml">Traditional ML</option>
        <option value="bert">Deep Learning (BERT)</option>
      </select>
      <br />
      <button onClick={detectFakeNews}>Detect</button>
      <h3>Result: {result}</h3>
    </div>
  );
}

export default App;