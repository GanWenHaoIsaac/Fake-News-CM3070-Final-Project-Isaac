import { useState } from "react";
import axios from "axios";

const FakeNewsInput = () => {
    const [text, setText] = useState("");
    const [model, setModel] = useState("lr");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [confidence, setConfidence] = useState(null);
    const models = ["lr", "dt", "svm", "nb", "rf", "lstm", "cnn-lstm", "bert", "bert-lstm"];

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setResult(null);
        setConfidence(null);

        try {
            const response = await axios.post("http://127.0.0.1:5000/predict", {
                text,
                model
            });

            setResult(response.data.prediction);
            setConfidence(response.data.confidence);
        } catch (error) {
            console.error("Error making prediction:", error);
            setResult("Error during prediction.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container mx-auto p-5">
            <h2 className="text-2xl font-bold mb-4">Fake News Detector</h2>
            <form onSubmit={handleSubmit} className="bg-white p-4 shadow-md rounded-lg">
                <textarea
                    className="w-full p-2 border rounded-md"
                    rows="5"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Enter news article..."
                    required
                /><br/>
                <label className="block mt-4">Choose a fake news detection model:</label><br></br>
                <select className="w-full p-2 border rounded-md" value={model} onChange={(e) => setModel(e.target.value)}>
                    {models.map((m) => (
                        <option key={m} value={m}>
                            {m.toUpperCase()}
                        </option>
                    ))}
                </select><br/>
                <button
                    type="submit"
                    className="bg-blue-500 text-white px-4 py-2 rounded-md mt-4 w-full"
                    disabled={loading}
                >
                    {loading ? "Analyzing..." : "Check"}
                </button>
            </form>

            {loading && (
                <div className="mt-4">
                    <div className="spinner border-t-4 border-blue-500 rounded-full w-8 h-8 animate-spin mx-auto"></div>
                    <p className="text-center">Analyzing article... Please wait...</p>
                </div>
            )}

            {result && (
                <div className="mt-4 p-4 bg-gray-100 rounded-md">
                    <h3 className="text-lg font-semibold">Prediction: {result}</h3>
                    <p className="text-gray-600 italic">Confidence: {confidence}</p>
                </div>
            )}
        </div>
    );
};

export default FakeNewsInput;
