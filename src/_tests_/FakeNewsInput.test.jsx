import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, test, expect, vi } from "vitest";
import FakeNewsInput from "../components/FakeNewsInput"; 
import axios from "axios";
import "@testing-library/jest-dom";


vi.mock("axios"); // Mocking axios to avoid real API calls

describe("FakeNewsInput Component", () => {
    test("renders form elements correctly", () => {
        render(<FakeNewsInput />);
        
        expect(screen.getByPlaceholderText("Enter news article...")).toBeInTheDocument();
        expect(screen.getByText("Choose a fake news detection model:")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Check" })).toBeInTheDocument();
    });

    test("updates textarea and select input", () => {
        render(<FakeNewsInput />);
        
        const textarea = screen.getByPlaceholderText("Enter news article...");
        fireEvent.change(textarea, { target: { value: "Breaking news..." } });
        expect(textarea.value).toBe("Breaking news...");

        const select = screen.getByRole("combobox");
        fireEvent.change(select, { target: { value: "bert" } });
        expect(select.value).toBe("bert");
    });

    test("submits form and displays result", async () => {
        axios.post.mockResolvedValue({ data: { prediction: "Fake", confidence: "0.95" } });

        render(<FakeNewsInput />);
        
        fireEvent.change(screen.getByPlaceholderText("Enter news article..."), { target: { value: "Breaking news..." } });

        fireEvent.click(screen.getByRole("button", { name: /check/i })); 
        await waitFor(() => {
            expect(screen.getByRole("button", { name: /analyzing/i })).toBeDisabled();
        });

        await waitFor(() => {
            expect(screen.getByText("Prediction: Fake")).toBeInTheDocument();
            expect(screen.getByText("Confidence: 0.95")).toBeInTheDocument();
        });
    });

    test("handles API error", async () => {
        axios.post.mockRejectedValue(new Error("Network Error"));

        render(<FakeNewsInput />);
        
        fireEvent.change(screen.getByPlaceholderText("Enter news article..."), { target: { value: "Breaking news..." } });
        fireEvent.click(screen.getByRole("button", { name: "Check" }));

        await waitFor(() => {
            expect(screen.getByText((content) => content.includes("Error during prediction"))).toBeInTheDocument();
        });
    });

    test("shows loading spinner when submitting", async () => {
        axios.post.mockResolvedValue({ data: { prediction: "Fake", confidence: "0.95" } });
    
        render(<FakeNewsInput />);
        
        fireEvent.change(screen.getByPlaceholderText("Enter news article..."), { target: { value: "Breaking news..." } });
        fireEvent.click(screen.getByRole("button", { name: /check/i }));
    
        await waitFor(() => {
            expect(screen.getByText("Analyzing article... Please wait...")).toBeInTheDocument();
        });
    });
    
    test("toggles sidebar visibility", () => {
        render(<FakeNewsInput />);
    
        const toggleButton = screen.getByRole("button", { name: /open menu/i });
    
        // Sidebar should initially be hidden
        expect(screen.getByText("Sample Texts")).not.toBeVisible();
    
        fireEvent.click(toggleButton);
    
        // Sidebar should now be visible
        expect(screen.getByText("Sample Texts")).toBeVisible();
    
        fireEvent.click(toggleButton);
    
        // Sidebar should be hidden again
        expect(screen.getByText("Sample Texts")).not.toBeVisible();
    });

    test("fetches and displays history", async () => {
        axios.get.mockResolvedValue({
            data: {
                history: [
                    { model: "bert", prediction: "Real", confidence: "0.92", text: "News text" },
                    { model: "bert", prediction: "Fake", confidence: "0.85", text: "News text" },
                ],
            },
        });
    
        render(<FakeNewsInput />);

        await waitFor(() => {
            const bertEntries = screen.getAllByText((content) => content.includes("BERT"));
            expect(bertEntries.length).toBeGreaterThan(0); 
    
            const realEntries = screen.getAllByText((content) => content.includes("Real"));
            expect(realEntries.length).toBeGreaterThan(0); 
    
            const confidenceEntries = screen.getAllByText((content) => content.includes("0.92"));
            expect(confidenceEntries.length).toBeGreaterThan(0); 
        });
    });

    test("copies text to clipboard", async () => {
    
        Object.defineProperty(navigator, "clipboard", {
            value: {
                writeText: vi.fn().mockResolvedValue(undefined),
            },
            writable: true,
        });

        render(<FakeNewsInput />);

        const copyButtons = screen.getAllByTestId("copy-button");
        expect(copyButtons.length).toBeGreaterThan(0);
        fireEvent.click(copyButtons[0]);
    
        await waitFor(() => {
            expect(navigator.clipboard.writeText).toHaveBeenCalledWith(expect.any(String));
        });
    });
     
});
