"""
Mistake Detection and Evaluation System

This module evaluates agent executions and detects various types of mistakes.
"""

from typing import List, Dict
from memory import ExecutionTrace, MistakeType
from langchain_core.messages import AIMessage, ToolMessage


class ExecutionEvaluator:
    """
    Evaluates agent executions and detects mistakes.
    """

    # Define expected tool sequences for travel planning
    REQUIRED_TOOLS = ["check_weather"]
    RECOMMENDED_SEQUENCE = [
        "check_weather",
        "search_flights",
        "recommend_hotels",
        "create_itinerary"
    ]

    def __init__(self):
        pass

    def evaluate(self, trace: ExecutionTrace, messages: List) -> ExecutionTrace:
        """
        Evaluate an execution trace and detect mistakes.

        Args:
            trace: Execution trace to evaluate
            messages: List of messages from the agent execution

        Returns:
            Updated trace with mistakes detected
        """
        # Extract tool calls from messages
        self._extract_tool_calls(trace, messages)

        # Check for various types of mistakes
        self._check_missing_required_tools(trace)
        self._check_tool_sequence(trace)
        self._check_early_termination(trace)
        self._check_tool_output_usage(trace, messages)

        # Determine success
        trace.success = len(trace.mistakes) == 0

        return trace

    def _extract_tool_calls(self, trace: ExecutionTrace, messages: List):
        """Extract tool calls from messages."""
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call["name"]
                    arguments = tool_call.get("args", {})

                    # Find corresponding output
                    output = ""
                    for i, m in enumerate(messages):
                        if isinstance(m, ToolMessage) and m.tool_call_id == tool_call["id"]:
                            output = m.content
                            break

                    trace.add_tool_call(tool_name, arguments, output)

            # Extract final answer
            if isinstance(msg, AIMessage) and msg.content:
                # Check if this is a final answer (no tool calls)
                if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                    trace.set_final_answer(msg.content)

    def _check_missing_required_tools(self, trace: ExecutionTrace):
        """Check if required tools were used."""
        tools_used = {call["tool"] for call in trace.tool_calls}

        for required_tool in self.REQUIRED_TOOLS:
            if required_tool not in tools_used:
                trace.add_mistake(
                    MistakeType.MISSING_REQUIRED_TOOL,
                    f"Required tool '{required_tool}' was not used",
                    step=None
                )

    def _check_tool_sequence(self, trace: ExecutionTrace):
        """Check if tools were called in the correct sequence."""
        if len(trace.tool_calls) < 2:
            return

        tools_used = [call["tool"] for call in trace.tool_calls]

        # Check for common sequence violations
        # 1. Hotels before flights
        if "recommend_hotels" in tools_used and "search_flights" in tools_used:
            hotel_idx = tools_used.index("recommend_hotels")
            flight_idx = tools_used.index("search_flights")

            if hotel_idx < flight_idx:
                trace.add_mistake(
                    MistakeType.WRONG_SEQUENCE,
                    "Hotels were recommended before searching for flights",
                    step=hotel_idx + 1
                )

        # 2. Itinerary before other tools
        if "create_itinerary" in tools_used:
            itinerary_idx = tools_used.index("create_itinerary")

            # Itinerary should be last
            if itinerary_idx < len(tools_used) - 1:
                trace.add_mistake(
                    MistakeType.WRONG_SEQUENCE,
                    "Itinerary was created before completing other planning steps",
                    step=itinerary_idx + 1
                )

        # 3. Weather should be checked first
        if "check_weather" in tools_used:
            weather_idx = tools_used.index("check_weather")
            if weather_idx > 0:
                trace.add_mistake(
                    MistakeType.WRONG_SEQUENCE,
                    "Weather should be checked before other tools",
                    step=weather_idx + 1
                )

    def _check_early_termination(self, trace: ExecutionTrace):
        """Check if agent terminated too early."""
        # If we have a final answer but didn't use minimum expected tools
        if trace.final_answer and len(trace.tool_calls) < 2:
            trace.add_mistake(
                MistakeType.TOO_EARLY_ANSWER,
                f"Agent provided final answer after only {len(trace.tool_calls)} tool calls",
                step=len(trace.tool_calls)
            )

    def _check_tool_output_usage(self, trace: ExecutionTrace, messages: List):
        """Check if tool outputs were properly used."""
        # Check if the final answer is generic and doesn't reference tool outputs
        if not trace.final_answer:
            return

        final_answer_lower = trace.final_answer.lower()

        # Look for indicators that tool outputs were ignored
        generic_phrases = [
            "i cannot",
            "i don't have access",
            "i'm unable to",
            "i can't help"
        ]

        # If we used tools but the answer is generic
        if len(trace.tool_calls) > 0:
            is_generic = any(phrase in final_answer_lower for phrase in generic_phrases)

            if is_generic:
                trace.add_mistake(
                    MistakeType.IGNORED_TOOL_OUTPUT,
                    "Agent ignored tool outputs and provided generic response",
                    step=len(trace.tool_calls) + 1
                )

    def print_evaluation(self, trace: ExecutionTrace):
        """Print a human-readable evaluation."""
        print(f"\n{'='*70}")
        print(f"EVALUATION - Run #{trace.run_id}")
        print(f"{'='*70}")
        print(f"Task: {trace.task}")
        print(f"Timestamp: {trace.timestamp}")
        print(f"Success: {'✓' if trace.success else '✗'}")
        print(f"\nTools Used ({len(trace.tool_calls)}):")

        for i, call in enumerate(trace.tool_calls, 1):
            print(f"  {i}. {call['tool']}")
            if call['arguments']:
                print(f"     Args: {call['arguments']}")

        if trace.mistakes:
            print(f"\nMistakes Detected ({len(trace.mistakes)}):")
            for i, mistake in enumerate(trace.mistakes, 1):
                step_info = f" [Step {mistake['step']}]" if mistake['step'] else ""
                print(f"  {i}. [{mistake['type']}]{step_info}")
                print(f"     {mistake['description']}")
        else:
            print("\n✓ No mistakes detected!")

        if trace.final_answer:
            print(f"\nFinal Answer Preview:")
            preview = trace.final_answer[:200] + "..." if len(trace.final_answer) > 200 else trace.final_answer
            print(f"  {preview}")

        print(f"{'='*70}\n")
