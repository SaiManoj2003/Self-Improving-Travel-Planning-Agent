# Quick Start Guide

Get the self-improving agent running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up GROQ API Key

Create a `.env` file:

```bash
# On Linux/Mac
cp .env.example .env

# On Windows
copy .env.example .env
```

Edit `.env` and add your API key:
```
GROQ_API_KEY=sk-your-actual-key-here
```

Get your API key from: https://console.groq.com/keys

## Step 3: Run the Demonstration

```bash
python main.py
```

This will:
- Run 10 iterations of the travel planning agent
- Show mistakes being detected
- Display learning in real-time
- Print improvement metrics

## What to Expect

### Early Runs (1-3)
You'll see mistakes like:
```
❌ Mistake: Required tool 'check_weather' was not used
❌ Mistake: Hotels were recommended before searching for flights
```

### After Learning
You'll see constraints being created:
```
LEARNED CONSTRAINTS (Active Reminders):
1. ALWAYS use the required tool mentioned: check_weather
```

## Alternative: Run Single Task

Test with one specific task:

```bash
python main.py single "Plan a 3-day trip to Barcelona"
```
Enjoy exploring the self-improving agent!
