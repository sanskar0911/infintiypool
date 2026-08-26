# Conversation State Memory Implementation

**Milestone Target:** 29-Aug-2026  
**Topic:** Conversation State & Short-Term Memory  
**Project:** Equity Strategy Backtesting Engine (Shankh / Decuple Internship 2026-27 Project TE1)  

---

## 1. Executive Overview

The memory module (`src/rag/memory.py`) provides lightweight conversational context tracking for the presentation UI. It allows the RAG assistant to resolve pronoun references in follow-up user questions (such as *"What are its limitations?"*) while supporting an explicit `[Clear Memory]` state reset action.

---

## 2. Multi-Turn Interaction Walkthrough

### Turn 1:
- **User**: *"Explain Simple Moving Average (SMA) crossover strategy."*
- **Assistant**: *Provides detailed explanation of SMA crossover signals, Golden Cross, and Death Cross.*
- **Memory Logged**: User query added to state history.

### Turn 2 (Follow-up with pronoun reference):
- **User**: *"What are its limitations?"*
- **Memory Context Resolver**: Detects pronoun `"its"` and resolves context topic to Turn 1 (`"Simple Moving Average (SMA)"`).
- **Resolved Query**: *"What are its limitations? (Context topic: Explain Simple Moving Average (SMA) crossover strategy.)"*
- **Assistant**: *Provides focused breakdown of SMA limitations (lags trends, whipsaw losses in choppy markets).*

### Reset Action:
- **User Action**: Clicks **`[Clear Memory]`** button in Streamlit UI.
- **System Action**: State history array reset to `[]`.
