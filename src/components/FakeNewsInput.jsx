import { useState, useEffect } from "react";
import axios from "axios";

const FakeNewsInput = () => {
    const [text, setText] = useState("");
    const [model, setModel] = useState("lr");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [confidence, setConfidence] = useState(null);
    const [history, setHistory] = useState([]);
    const [isSidebarOpen, setIsSidebarOpen] = useState(false); // Sidebar state
    const models = ["lr", "dt", "svm", "nb", "rf", "lstm", "cnn-lstm", "bert", "bert-lstm"];

    const fakeExamples = [
        "Shocking! Scientists confirm the earth is flat!",
        "Exclusive: Secret government agency controls all world leaders.",
        "New study reveals drinking soda makes you immune to diseases.",
        "Alien spaceship spotted near the White House!",
        "You won’t believe this miracle cure for aging!"
    ];

    const realExamples = [
        "Head of a conservative Republican faction in the U.S. Congress urged budget restraint in 2019",
        "Latest update: Scientists discover a new exoplanet.",
        "Health news: Regular exercise improves mental health.",
        "Technology: AI is transforming the future of work.",
        "World news: Climate change policies gain momentum."
    ];

    const MAX_LENGTH = 70; 

    useEffect(() => {
        fetchHistory();
    }, []);

    const fetchHistory = async () => {
        try {
            const response = await axios.get("http://127.0.0.1:5000/history");
            setHistory(response.data.history);
        } catch (error) {
            console.error("Error fetching history:", error);
        }
    };

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
            fetchHistory();
        } catch (error) {
            console.error("Error making prediction:", error);
            setResult("Error during prediction.");
        } finally {
            setLoading(false);
        }
    };

    const handleCopy = (textToCopy) => {
        navigator.clipboard.writeText(textToCopy)
            .then(() => alert("Copied to clipboard!"))
            .catch(err => console.error("Error copying text:", err));
    };

    return (
        <div className="container mx-auto p-5 relative">
            {/* Sidebar Toggle Button */}
            <button
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                className="fixed top-5 left-5 bg-gray-700 text-white px-3 py-2 rounded-md"
            >
                {isSidebarOpen ? "Close Menu" : "Open Menu"}
            </button>

            {/* Sidebar Menu */}
            <div 
                className={`fixed top-0 left-0 h-full w-64 bg-gray-800 text-white p-5 z-50 shadow-lg 
                transition-transform duration-300 ease-in-out ${isSidebarOpen ? "translate-x-0" : "-translate-x-full"}`}
                style={{ display: isSidebarOpen ? "block" : "none" }} // Ensures it's hidden when closed
            >
                <h3 className="text-xl font-bold mb-4">Sample Texts</h3>
                {/* Fake News Examples */}
                <h4 className="text-lg font-semibold mt-4 mb-2 text-red-400">Fake News Examples</h4>
                <ul>
                    {fakeExamples.map((sample, index) => (
                        <li key={index} className="mb-3 p-2 bg-red-700 rounded-md flex justify-between items-center">
                            <span className="text-black">
                            {sample.length > MAX_LENGTH ? sample.substring(0, MAX_LENGTH) + "........" : sample}
                            </span>
                            <button 
                                onClick={() => handleCopy(sample)}
                                className="ml-4 bg-gray-400 hover:bg-gray-500 text-black font-semibold py-1 px-2 rounded"
                            >Copy</button>
                        </li>
                    ))}
                </ul>

                {/* Real News Examples */}
                <h4 className="text-lg font-semibold mt-4 mb-2 text-green-400">Real News Examples</h4>
                <ul>
                    {realExamples.map((sample, index) => (
                        <li key={index} className="mb-3 p-2 bg-green-700 rounded-md flex justify-between items-center">
                            <span className="text-black">
                            {sample.length > MAX_LENGTH ? sample.substring(0, MAX_LENGTH) + "..." : sample}
                            </span>
                            <button 
                                onClick={() => handleCopy(sample)}
                                className="bg-gray-400 hover:bg-gray-500 text-black font-semibold py-1 px-2 rounded"
                            >
                                Copy
                            </button>
                        </li>
                    ))}
                </ul>
            </div>


            <h2 className="text-2xl font-bold mb-4">Fake News Detector</h2>
            <form onSubmit={handleSubmit} className="bg-white p-4 shadow-md rounded-lg">
                <textarea
                    className="w-full p-2 border rounded-md"
                    rows="5"
                    cols="50"
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
                    <p className="text-gray-600 italic">Confidence: {confidence}</p><br></br>
                </div>
            )}
            <h3 className="text-xl font-bold mt-6">History</h3>
            <div className="mt-2 bg-white p-4 shadow-md rounded-lg">
                {history.length === 0 ? (
                    <p className="text-gray-600">No history available.</p>
                ) : (
                    <ul className="list-disc pl-5">
                        {history.map((entry, index) => (
                            <li key={index} className="mb-2">
                                <strong>{entry.model.toUpperCase()}:</strong> {entry.prediction} ({entry.confidence})
                                <br></br><strong>{entry.text}</strong><br></br><br></br>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
};

export default FakeNewsInput;
