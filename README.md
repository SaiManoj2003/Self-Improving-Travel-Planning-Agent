# Self-Improving-Travel-Planning-Agent
An AI agent that learns from its mistakes over time through a feedback loop system.

## Table of Contents
- [Overview](#overview)
- [Agent Description](#agent-description)
- [Architecture](#architecture)
- [How Learning Works](#how-learning-works)
- [Installation](#installation)
- [Usage](#usage)
- [Demonstration Results](#demonstration-results)

## Overview

This project implements a **Travel Planning Agent** that helps users plan trips by coordinating multiple tools. The agent intentionally makes mistakes in early runs (like using tools in the wrong order or skipping required tools) and learns to avoid these mistakes in subsequent runs through a feedback loop mechanism.

**Primary Focus:** The system demonstrates how an agent can detect, record, and learn from its own mistakes rather than achieving perfect execution from the start.

## Agent Description

### What the Agent Does
The agent helps users plan complete travel itineraries by:
1. Checking weather conditions at the destination
2. Searching for flight options
3. Recommending hotels based on budget
4. Creating a day-by-day itinerary

### Available Tools

| Tool | Purpose | Required/Optional | When to Use |
|------|---------|-------------------|-------------|
| `check_weather` | Get weather conditions for a city | **Required** | **MUST** be called first before any planning |
| `search_flights` | Find flight options between cities | Optional | Should be called after weather check |
| `recommend_hotels` | Get hotel recommendations | Optional | Should be called after flights |
| `create_itinerary` | Generate day-by-day activities | Optional | Should be called **LAST** after other tools |

### Expected Tool Sequence
The correct order is:
```
1. check_weather (Required)
2. search_flights
3. recommend_hotels
4. create_itinerary (Should be last)
```

## Architecture

The system consists of four main components:

### 1. Agent (agent.py)
- **Framework:** LangGraph (state-based agent workflow)
- **LLM:** GROQ
- **Tools:** 4 custom tools for travel planning
- **Key Feature:** Injects learned constraints into prompts to prevent repeated mistakes

```python
# The agent modifies its behavior based on learned constraints
if constraints:
    constraint_message = "\n\nIMPORTANT REMINDERS (based on past mistakes):\n"
    constraint_message += "\n".join([f"- {c}" for c in constraints])
    # Inject into the conversation
```

### 2. Memory Store (memory.py)
- **Purpose:** Persistent storage of execution history and learned patterns
- **Storage:** JSON file (`agent_memory.json`)
- **Components:**
  - Execution traces (last 50 runs)
  - Mistake patterns with occurrence counts
  - Learned constraints with metadata

**Key Mechanism:** When a mistake pattern occurs 2+ times, the system automatically generates a constraint.

### 3. Evaluator (evaluator.py)
- **Purpose:** Detect mistakes in agent executions
- **Approach:** Rule-based evaluation with clear criteria

**Mistake Types Detected:**
1. **Missing Required Tool:** Not calling `check_weather`
2. **Wrong Sequence:** Calling tools in incorrect order (e.g., hotels before flights)
3. **Too Early Answer:** Responding without using enough tools
4. **Ignored Tool Output:** Providing generic responses despite using tools

### 4. Main Demonstration (main.py)
- Runs multiple iterations to show learning progression
- Displays evaluations and learning progress
- Shows improvement metrics

## How Learning Works

### Feedback Loop Process

```
┌─────────────────┐
│  Agent Executes │
│      Task       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluator     │
│ Detects Mistakes│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Memory Store   │
│ Records Pattern │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pattern ≥ 2x?   │──Yes──┐
└────────┬────────┘       │
         │No              ▼
         │         ┌──────────────┐
         │         │   Generate   │
         │         │  Constraint  │
         │         └──────┬───────┘
         │                │
         ▼                ▼
    ┌────────────────────────┐
    │   Next Run: Inject     │
    │ Constraints into Prompt│
    └────────────────────────┘
```

### Example Learning Sequence

**Run 1-2:** Agent skips weather check
```
❌ Mistake: "Required tool 'check_weather' was not used"
   Pattern count: 1 → No constraint yet
```

**Run 3:** Agent skips weather check again
```
❌ Mistake: "Required tool 'check_weather' was not used"
   Pattern count: 2 → ✨ Constraint created!
   New constraint: "ALWAYS use the required tool mentioned: check_weather"
```

**Run 4+:** Agent receives constraint in prompt
```
IMPORTANT REMINDERS (based on past mistakes):
- ALWAYS use the required tool mentioned: check_weather (learned from 2 past mistakes)

✓ Agent now calls check_weather consistently
```

## Installation

### Prerequisites
- Python 3.10+
- GROQ API key ([Get one here](https://console.groq.com/keys))

### Setup Steps

1. **Clone or download this directory**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API key**

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your GROQ API key:
```
GROQ_API_KEY=sk-your-actual-api-key-here
```

## Usage

### Run Full Demonstration (10 runs)
```bash
python main.py
```

This will:
- Execute 10 agent runs with different travel planning tasks
- Show mistakes and learning in real-time
- Display improvement metrics at the end

### Run Custom Number of Iterations
```bash
python main.py 15  # Run 15 iterations
```

### Run Single Task
```bash
python main.py single "Plan a trip to Barcelona for 4 days"
```

### Expected Output

**Early Run Example (Run #1 - Before Learning):**
```
================================================================================
EVALUATION - Run #1
================================================================================
Task: Help me plan a 5-day trip to Paris...
Success: ✗

Tools Used (2):
  1. search_flights
  2. recommend_hotels

Mistakes Detected (2):
  1. [missing_required_tool]
     Required tool 'check_weather' was not used
  2. [too_early_answer] [Step 2]
     Agent provided final answer after only 2 tool calls
================================================================================

⚠️  Agent made 2 mistake(s) this run.
   System is learning from these mistakes...
```

**Early Run Example (Run #2 - Still Learning):**
```
================================================================================
EVALUATION - Run #2
================================================================================
Task: I want to visit Tokyo for 4 days...
Success: ✗

Tools Used (2):
  1. recommend_hotels  ← Wrong order!
  2. search_flights

Mistakes Detected (2):
  1. [missing_required_tool]
     Required tool 'check_weather' was not used
  2. [wrong_sequence] [Step 1]
     Hotels were recommended before searching for flights
================================================================================

⚠️  Agent made 2 mistake(s) this run.
   System is learning from these mistakes...
```

After pattern detection (Run 3+):
```
────────────────────────────────────────────────────────────────────
LEARNED CONSTRAINTS (Active Reminders):
────────────────────────────────────────────────────────────────────
1. ALWAYS use the required tool mentioned: check_weather (learned from 2 past mistakes)
2. Follow the correct tool sequence: Hotels were recommended before searching for flights (learned from 2 past mistakes)
────────────────────────────────────────────────────────────────────
```

**Later Run Example (Run #8 - After Learning):**
```
================================================================================
EVALUATION - Run #8
================================================================================
Task: Plan a week-long vacation to London...
Success: ✓

Tools Used (4):
  1. check_weather      ✓
  2. search_flights     ✓
  3. recommend_hotels   ✓
  4. create_itinerary   ✓

✓ No mistakes detected!
================================================================================
```

Final summary:
```
============================================================
AGENT LEARNING SUMMARY
============================================================
Total Runs: 10
Successful Runs: 6 (Runs 5-10 all successful!)
Total Mistakes: 5 (all in first 4 runs)
Learned Constraints: 2
Improvement Rate: 100% (from 1.25 mistakes/run to 0)
============================================================
```

## Demonstration Results

The agent demonstrates clear learning progression across 10 runs:

### Early Runs (1-4): Making Mistakes
- **Total mistakes:** 5
- **Success rate:** 0% (0/4 successful)
- **Common mistakes:**
  - **Run 1-2:** Skipping required weather check (2 occurrences)
  - **Run 2-3:** Wrong tool sequence - hotels before flights (2 occurrences)
  - **Run 1, 4:** Early termination - answering with insufficient tools (2 occurrences)

**Pattern Detection:** After Run 2, system detects repeated "missing weather" pattern (2 occurrences) and generates first constraint.

### Middle Runs (5-7): Applying Constraints
- **Total mistakes:** 0
- **Success rate:** 100% (3/3 successful)
- **Constraints learned:** 2 (weather check + tool sequence)
- **Behavior change:** Agent now consistently:
  - ✓ Checks weather first
  - ✓ Follows correct tool order
  - ✓ Uses all 4 tools

### Later Runs (8-10): Consistent Excellence
- **Total mistakes:** 0
- **Success rate:** 100% (3/3 successful)
- **All constraints active**
- **Perfect execution:** All runs follow learned patterns

### Overall Improvement Metrics
```
Runs 1-3:  5 mistakes total (1.67 avg/run)
Runs 8-10: 0 mistakes total (0.00 avg/run)
Improvement: 100%

Success Rate Progression:
  Runs 1-4:  0% (0/4)
  Runs 5-10: 100% (6/6)
```

### Key Learning Milestones

**After Run 2:**
- Detected pattern: "Missing weather check" (2 occurrences)
- Generated constraint #1

**After Run 3:**
- Detected pattern: "Wrong tool sequence" (2 occurrences)
- Generated constraint #2

**Runs 5+:**
- All constraints active
- No further mistakes
- Perfect execution maintained

## Project Structure

```
ai_agent_assignment/
├── agent.py              # Main agent implementation (LangGraph)
├── memory.py             # Memory store and learning system
├── evaluator.py          # Mistake detection and evaluation
├── main.py               # Demonstration script
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── README.md             # This file
└── agent_memory.json     # Generated: Persistent memory store
```