"""
Memory and Learning System for Self-Improving Agent

This module handles:
1. Recording execution traces
2. Detecting mistakes
3. Learning from patterns
4. Storing and retrieving learned constraints
"""

import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict


class MistakeType:
    """Types of mistakes the agent can make."""
    MISSING_REQUIRED_TOOL = "missing_required_tool"
    WRONG_TOOL = "wrong_tool"
    WRONG_SEQUENCE = "wrong_sequence"
    TOO_EARLY_ANSWER = "too_early_answer"
    IGNORED_TOOL_OUTPUT = "ignored_tool_output"


class ExecutionTrace:
    """Represents a single execution trace."""

    def __init__(self, run_id: int, task: str, timestamp: str):
        self.run_id = run_id
        self.task = task
        self.timestamp = timestamp
        self.tool_calls = []
        self.final_answer = ""
        self.success = False
        self.mistakes = []

    def add_tool_call(self, tool_name: str, arguments: dict, output: str):
        """Add a tool call to the trace."""
        self.tool_calls.append({
            "tool": tool_name,
            "arguments": arguments,
            "output": output,
            "order": len(self.tool_calls) + 1
        })

    def set_final_answer(self, answer: str):
        """Set the final answer."""
        self.final_answer = answer

    def add_mistake(self, mistake_type: str, description: str, step: Optional[int] = None):
        """Add a detected mistake."""
        self.mistakes.append({
            "type": mistake_type,
            "description": description,
            "step": step,
            "timestamp": datetime.now().isoformat()
        })

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "task": self.task,
            "timestamp": self.timestamp,
            "tool_calls": self.tool_calls,
            "final_answer": self.final_answer,
            "success": self.success,
            "mistakes": self.mistakes
        }


class MemoryStore:
    """
    Memory store for learning from past mistakes.
    """

    def __init__(self, storage_path: str = "agent_memory.json"):
        self.storage_path = Path(storage_path)
        self.execution_history: List[ExecutionTrace] = []
        self.learned_constraints: List[Dict] = []
        self.mistake_patterns: Dict[str, int] = defaultdict(int)
        self.run_counter = 0

        # Load existing memory
        self._load()

    def _load(self):
        """Load memory from disk."""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.run_counter = data.get("run_counter", 0)
                self.learned_constraints = data.get("learned_constraints", [])
                self.mistake_patterns = defaultdict(int, data.get("mistake_patterns", {}))

                # Load execution history (last 50 runs)
                history_data = data.get("execution_history", [])
                for trace_data in history_data[-50:]:
                    trace = ExecutionTrace(
                        run_id=trace_data["run_id"],
                        task=trace_data["task"],
                        timestamp=trace_data["timestamp"]
                    )
                    trace.tool_calls = trace_data["tool_calls"]
                    trace.final_answer = trace_data["final_answer"]
                    trace.success = trace_data["success"]
                    trace.mistakes = trace_data["mistakes"]
                    self.execution_history.append(trace)

    def _save(self):
        """Save memory to disk."""
        data = {
            "run_counter": self.run_counter,
            "learned_constraints": self.learned_constraints,
            "mistake_patterns": dict(self.mistake_patterns),
            "execution_history": [trace.to_dict() for trace in self.execution_history[-50:]]
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def create_trace(self, task: str) -> ExecutionTrace:
        """Create a new execution trace."""
        self.run_counter += 1
        trace = ExecutionTrace(
            run_id=self.run_counter,
            task=task,
            timestamp=datetime.now().isoformat()
        )
        return trace

    def save_trace(self, trace: ExecutionTrace):
        """Save an execution trace."""
        self.execution_history.append(trace)

        # Update mistake patterns
        for mistake in trace.mistakes:
            pattern_key = f"{mistake['type']}:{mistake['description']}"
            self.mistake_patterns[pattern_key] += 1

        # Learn from patterns (trigger after 2 occurrences)
        self._learn_from_patterns()

        # Persist to disk
        self._save()

    def _learn_from_patterns(self):
        """Analyze patterns and create new constraints."""
        for pattern_key, count in self.mistake_patterns.items():
            # If a mistake happens 2+ times, create a constraint
            if count >= 2:
                mistake_type, description = pattern_key.split(":", 1)

                # Check if we already have a constraint for this
                existing = any(
                    c["pattern_key"] == pattern_key
                    for c in self.learned_constraints
                )

                if not existing:
                    constraint = self._create_constraint(mistake_type, description, count)
                    if constraint:
                        self.learned_constraints.append({
                            "pattern_key": pattern_key,
                            "constraint": constraint,
                            "occurrences": count,
                            "created_at": datetime.now().isoformat()
                        })

    def _create_constraint(self, mistake_type: str, description: str, count: int) -> Optional[str]:
        """Create a constraint based on mistake type."""
        constraints_map = {
            MistakeType.MISSING_REQUIRED_TOOL:
                f"ALWAYS use the required tool mentioned: {description}",
            MistakeType.WRONG_SEQUENCE:
                f"Follow the correct tool sequence: {description}",
            MistakeType.TOO_EARLY_ANSWER:
                "Do NOT provide a final answer until ALL necessary tools have been called",
            MistakeType.IGNORED_TOOL_OUTPUT:
                f"MUST incorporate tool outputs into your answer: {description}",
            MistakeType.WRONG_TOOL:
                f"Use the correct tool: {description}"
        }

        # Generate appropriate constraint
        if mistake_type in constraints_map:
            base_constraint = constraints_map[mistake_type]
            return f"{base_constraint} (learned from {count} past mistakes)"

        return None

    def get_active_constraints(self) -> List[str]:
        """Get all active learned constraints."""
        return [c["constraint"] for c in self.learned_constraints]

    def get_statistics(self) -> dict:
        """Get learning statistics."""
        total_runs = len(self.execution_history)
        successful_runs = sum(1 for trace in self.execution_history if trace.success)
        total_mistakes = sum(len(trace.mistakes) for trace in self.execution_history)

        # Calculate improvement rate (last 5 vs previous 5)
        if total_runs >= 10:
            recent_mistakes = sum(
                len(trace.mistakes)
                for trace in self.execution_history[-5:]
            )
            previous_mistakes = sum(
                len(trace.mistakes)
                for trace in self.execution_history[-10:-5]
            )
            improvement = (previous_mistakes - recent_mistakes) / max(previous_mistakes, 1) * 100
        else:
            improvement = 0

        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "total_mistakes": total_mistakes,
            "learned_constraints": len(self.learned_constraints),
            "improvement_rate": round(improvement, 2),
            "mistake_patterns": dict(self.mistake_patterns)
        }

    def print_summary(self):
        """Print a summary of the memory store."""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("AGENT LEARNING SUMMARY")
        print("="*60)
        print(f"Total Runs: {stats['total_runs']}")
        print(f"Successful Runs: {stats['successful_runs']}")
        print(f"Total Mistakes: {stats['total_mistakes']}")
        print(f"Learned Constraints: {stats['learned_constraints']}")
        print(f"Improvement Rate: {stats['improvement_rate']}%")

        if self.learned_constraints:
            print("\nLearned Constraints:")
            for i, constraint_data in enumerate(self.learned_constraints, 1):
                print(f"  {i}. {constraint_data['constraint']}")

        print("="*60 + "\n")
