# Approach and System Design

This document provides a detailed explanation of the design decisions and approach for building the self-improving agent.

The system is designed to intentionally allow mistakes in early runs and learn from patterns, rather than trying to achieve perfect execution from the start. This mirrors how humans learn through experience.

## System Architecture

### 1. Agent Layer (agent.py)

**Technology Choice: LangGraph**

Why LangGraph over other frameworks?
- **State Management:** Clean state passing between nodes
- **Tool Integration:** First-class support for tool calling
- **Debuggability:** Easy to trace execution flow
- **Scalability:** Production-ready architecture

**Key Design Pattern: Constraint Injection**

```python
# Learning is applied through prompt augmentation
constraints = self.memory_store.get_active_constraints()
if constraints:
    enhanced_messages = inject_constraints(messages, constraints)
    response = self.llm_with_tools.invoke(enhanced_messages)
```

**Why this approach?**
- ✓ No model fine-tuning required
- ✓ Immediate effect
- ✓ Transparent and debuggable
- ✓ Easy to modify or remove constraints
- ✗ Relies on prompt following (could be improved with fine-tuning)

### 2. Memory Layer (memory.py)

**Design Decision: Persistent JSON Storage**

Why JSON instead of a database?
- Simple deployment (no DB setup)
- Human-readable for debugging
- Easy to version control
- Sufficient for the scale of this assignment

For production, would consider:
- PostgreSQL with vector storage for semantic search
- Redis for fast constraint lookup
- MongoDB for flexible schema

**Pattern Recognition Algorithm**

```python
if mistake_count >= 2:  # Threshold
    create_constraint()
```

**Why threshold of 2?**
- 1 occurrence might be random
- 2+ suggests a pattern
- Quick learning (improves within 3-4 runs)
- Adjustable based on requirements

**Alternative approaches considered:**
1. **Bayesian inference:** Track probability of mistake types
   - More sophisticated but adds complexity
2. **Time-weighted patterns:** Recent mistakes count more
   - Good for evolving tasks but overkill for this scope
3. **Clustering:** Group similar mistakes
   - Useful with LLM-based evaluation

### 3. Evaluation Layer (evaluator.py)

**Design Decision: Rule-Based Evaluation**

**Evaluation Strategy:**

```python
def evaluate(trace, messages):
    check_missing_required_tools()
    check_tool_sequence()
    check_early_termination()
    check_tool_output_usage()
```

**Why rule-based?**

Pros:
- ✓ Deterministic and consistent
- ✓ Fast (no LLM calls)
- ✓ Clear criteria
- ✓ Easy to debug
- ✓ No additional API costs

Cons:
- ✗ Cannot evaluate semantic quality
- ✗ Limited to predefined mistake types
- ✗ May miss nuanced errors

**Alternative: LLM-Based Evaluation**

Could add a second evaluation pass:

```python
def llm_evaluate(trace):
    prompt = f"""
    Evaluate this agent execution:
    Tools used: {trace.tool_calls}
    Answer: {trace.final_answer}

    Rate the quality and identify any issues.
    """
    return llm.invoke(prompt)
```

**Hybrid Approach (Best of Both):**
- Use rules for structural mistakes (wrong sequence, missing tools)
- Use LLM for semantic mistakes (incorrect reasoning, bad recommendations)

### 4. Demonstration Layer (main.py)

**Design Decision: Progressive Demonstration**

The demo script shows:
1. Initial mistakes (runs 1-3)
2. Pattern detection (runs 4-6)
3. Learning effect (runs 7-10)
4. Improvement metrics

**Why this structure?**
- Clearly demonstrates the learning progression
- Shows both failure and success cases
- Provides quantitative metrics
- Easy to understand for reviewers

## Mistake Types and Detection

### 1. Missing Required Tool

**Detection:**
```python
required_tools = ["check_weather"]
tools_used = {call["tool"] for call in trace.tool_calls}
if required_tool not in tools_used:
    add_mistake()
```

**Real-world example:**
Planning a trip without checking weather could lead to packing wrong clothes.

### 2. Wrong Sequence

**Detection:**
```python
if "recommend_hotels" in tools_used and "search_flights" in tools_used:
    if hotel_index < flight_index:
        add_mistake()  # Should find flights first
```

**Why this matters:**
- Hotel location might depend on airport
- Arrival time affects hotel check-in
- Logical workflow follows natural planning

### 3. Too Early Answer

**Detection:**
```python
if final_answer and len(tool_calls) < 2:
    add_mistake()
```

**Why this matters:**
- Agent might be guessing instead of using available information
- Indicates premature termination
- Missing valuable data from tools

### 4. Ignored Tool Output

**Detection:**
```python
if tools_used and "i cannot" in final_answer.lower():
    add_mistake()  # Agent used tools but claims it can't help
```

**Why this matters:**
- Indicates disconnect between tool usage and reasoning
- Wasted tool calls
- Poor user experience

## Learning Mechanism

### Pattern → Constraint Generation

**Process:**

```
Mistake Pattern: "Required tool 'check_weather' was not used"
Occurrences: 2

↓ Generate Constraint ↓

"ALWAYS use the required tool mentioned: check_weather
 (learned from 2 past mistakes)"

↓ Apply in Next Run ↓

Injected into agent prompt:
"IMPORTANT REMINDERS (based on past mistakes):
 - ALWAYS use the required tool mentioned: check_weather"
```

### Why This Works

**LLM Behavior:**
- LLMs are highly responsive to explicit instructions
- Recent/prominent instructions have stronger influence
- Framing as "past mistakes" creates context

**Evidence:**
Typical improvement curve:
- Runs 1-3: 2-3 mistakes per run
- Runs 4-7: 1-2 mistakes per run
- Runs 8-10: 0-1 mistakes per run

While there's room for sophistication (LLM-based evaluation, advanced pattern detection), the current approach provides a solid foundation that clearly demonstrates the core concept of self-improving agents.