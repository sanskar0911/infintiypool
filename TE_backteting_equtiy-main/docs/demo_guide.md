# QUANTLAB Multilingual MADRID RAG Demo Guide

## How to Run the Application
1. Launch QUANTLAB Web Dashboard:
   ```bash
   python app.py
   ```
2. Open your web browser at: `http://localhost:8000`
3. Click on the **RAG Knowledge Base** tab in the main terminal interface.

## Interactive Demo Queries
Click any of the built-in demo query buttons:

1. **English Query**:
   `What is Sharpe ratio?`
2. **Hindi Query**:
   `Sharpe ratio kya hota hai?`
3. **Marathi Query**:
   `Sharpe ratio म्हणजे काय?`
4. **Spanish Query**:
   `¿Qué es el CAGR?`
5. **Marathi Max DD Query**:
   `Maximum drawdown म्हणजे काय?`

## What to Observe in the UI
1. **8 Connected Visual Stage Cards**: Observe automatic language detection, MADRID canonical representation, vector search latency, and decoded target output.
2. **Retrieved Knowledge Sources**: Expandable cards displaying source file paths, topic, section title, and similarity score %.
3. **Scientific Token Accounting Panel**: Breakdown of Original, MADRID, Context, LLM Input/Output, Final Output, and Observed Token Difference %.
