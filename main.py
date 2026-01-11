"""
Main demonstration script for the Self-Improving Travel Planning Agent

This script demonstrates:
1. Initial runs with mistakes
2. Learning from those mistakes
3. Improved performance in later runs
"""

import sys
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
from agent import TravelPlanningAgent
from memory import MemoryStore
from evaluator import ExecutionEvaluator
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time


def run_demonstration(num_runs: int = 10):
    """
    Run the agent multiple times to demonstrate learning.

    Args:
        num_runs: Number of runs to execute
    """
    print("\n" + "="*80)
    print("SELF-IMPROVING TRAVEL PLANNING AGENT - DEMONSTRATION")
    print("="*80)

    # Load environment variables
    load_dotenv()

    # Initialize components - Use Groq's LLM
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    # Initialize memory and evaluator
    memory_store = MemoryStore("agent_memory.json")
    evaluator = ExecutionEvaluator()

    # Initialize agent
    agent = TravelPlanningAgent(llm, memory_store)

    # Test tasks
    tasks = [
        "Help me plan a 5-day trip to Paris, France. I need to know about weather, flights from New York, hotels, and activities.",
        "I want to visit Tokyo for 4 days. Please help me plan everything including weather check, flights from Los Angeles, accommodation, and itinerary.",
        "Plan a week-long vacation to London. I'm traveling from Boston and need the complete travel plan.",
        "Organize a 3-day trip to Dubai. Check weather, find flights from Miami, suggest hotels, and create an itinerary.",
        "Help me plan a 6-day trip to Sydney, Australia from San Francisco. Full planning needed.",
    ]

    print(f"\nRunning {num_runs} iterations to demonstrate learning...\n")

    for i in range(num_runs):
        task = tasks[i % len(tasks)]

        print(f"\n{'#'*80}")
        print(f"# RUN {i + 1}/{num_runs}")
        print(f"{'#'*80}")

        # Create trace
        trace = memory_store.create_trace(task)

        try:
            # Run the agent
            print(f"\nTask: {task}\n")
            print("Agent is working...")

            result = agent.run(task)

            # Evaluate the run
            trace = evaluator.evaluate(trace, result["messages"])

            # Print evaluation
            evaluator.print_evaluation(trace)

            # Save to memory
            memory_store.save_trace(trace)

            # Show learning progress
            if trace.mistakes:
                print(f"⚠️  Agent made {len(trace.mistakes)} mistake(s) this run.")
                print("   System is learning from these mistakes...\n")
            else:
                print("✓ Perfect execution! No mistakes detected.\n")

            # Show active constraints after every few runs
            if (i + 1) % 3 == 0:
                constraints = memory_store.get_active_constraints()
                if constraints:
                    print("\n" + "─"*70)
                    print("LEARNED CONSTRAINTS (Active Reminders):")
                    print("─"*70)
                    for idx, constraint in enumerate(constraints, 1):
                        print(f"{idx}. {constraint}")
                    print("─"*70 + "\n")

            # Small delay for readability
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error during run {i + 1}: {str(e)}")
            trace.add_mistake("execution_error", str(e))
            memory_store.save_trace(trace)

    # Print final summary
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)

    memory_store.print_summary()

    # Show improvement metrics
    print("\nIMPROVEMENT ANALYSIS:")
    print("-"*80)

    if len(memory_store.execution_history) >= 6:
        first_three = memory_store.execution_history[:3]
        last_three = memory_store.execution_history[-3:]

        early_mistakes = sum(len(t.mistakes) for t in first_three)
        recent_mistakes = sum(len(t.mistakes) for t in last_three)

        print(f"First 3 runs: {early_mistakes} total mistakes")
        print(f"Last 3 runs: {recent_mistakes} total mistakes")

        if early_mistakes > 0:
            improvement = ((early_mistakes - recent_mistakes) / early_mistakes) * 100
            print(f"Improvement: {improvement:.1f}%")
        else:
            print("Perfect performance from the start!")

    print("-"*80)

    # Show example trace comparisons
    print("\nKEY INSIGHTS:")
    print("-"*80)

    if memory_store.learned_constraints:
        print(f"✓ Agent learned {len(memory_store.learned_constraints)} constraints from mistakes")
        print("✓ These constraints are now automatically applied to prevent future errors")
    else:
        print("✓ Agent performed well from the start")

    print("✓ Memory persists across runs in 'agent_memory.json'")
    print("✓ Agent improves autonomously without manual intervention")
    print("-"*80 + "\n")


def run_single_task(task: str):
    """
    Run a single task with the agent.

    Args:
        task: Task description
    """
    load_dotenv()

    # Initialize components
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    memory_store = MemoryStore("agent_memory.json")
    evaluator = ExecutionEvaluator()
    agent = TravelPlanningAgent(llm, memory_store)

    print("\n" + "="*80)
    print("SINGLE TASK EXECUTION")
    print("="*80)
    print(f"\nTask: {task}\n")

    # Create trace
    trace = memory_store.create_trace(task)

    # Run the agent
    result = agent.run(task)

    # Evaluate
    trace = evaluator.evaluate(trace, result["messages"])
    evaluator.print_evaluation(trace)

    # Save to memory
    memory_store.save_trace(trace)

    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys

    # Check if Groq API key is set
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("\n" + "!"*80)
        print("WARNING: GROQ_API_KEY not found in environment variables")
        print("!"*80)
        print("\nPlease set your Groq API key:")
        print("1. Create a .env file in this directory")
        print("2. Add: GROQ_API_KEY=your-api-key-here")
        print("\nGet a free API key from: https://console.groq.com/keys")
        print("\n" + "!"*80 + "\n")
        sys.exit(1)

    # Run demonstration
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Single task mode
        task = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else \
               "Help me plan a 5-day trip to Paris. Check weather, flights, hotels, and activities."
        run_single_task(task)
    else:
        # Full demonstration mode
        num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        run_demonstration(num_runs=num_runs)
