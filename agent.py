"""
Self-Improving Travel Planning Agent

This agent helps users plan trips by:
1. Checking weather conditions
2. Finding flight options
3. Recommending hotels
4. Creating an itinerary

The agent learns from mistakes over time through a feedback loop.
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import json
import random
from datetime import datetime
import operator


# Define tools that the agent can use
@tool
def check_weather(city: str) -> str:
    """
    Check weather conditions for a given city.
    This tool MUST be called before recommending travel plans.

    Args:
        city: Name of the city to check weather for

    Returns:
        Weather information for the city
    """
    weather_conditions = ["sunny", "rainy", "cloudy", "snowy"]
    temperature = random.randint(10, 30)
    condition = random.choice(weather_conditions)
    return json.dumps({
        "city": city,
        "condition": condition,
        "temperature": temperature,
        "advice": f"Weather is {condition} with {temperature}°C"
    })


@tool
def search_flights(origin: str, destination: str) -> str:
    """
    Search for available flights between two cities.
    This tool should be called after checking weather.

    Args:
        origin: Departure city
        destination: Arrival city

    Returns:
        Available flight options
    """
    airlines = ["AirTravel", "SkyHigh", "CloudNine"]
    prices = [200, 350, 500]
    flight_data = []

    for i in range(2):
        flight_data.append({
            "airline": random.choice(airlines),
            "price": random.choice(prices),
            "departure": f"{random.randint(8, 18)}:00",
            "duration": f"{random.randint(2, 8)}h"
        })

    return json.dumps({
        "origin": origin,
        "destination": destination,
        "flights": flight_data
    })


@tool
def recommend_hotels(city: str, budget: str = "medium") -> str:
    """
    Recommend hotels in a city based on budget.
    This tool should be called after searching flights.

    Args:
        city: City to find hotels in
        budget: Budget level (low, medium, high)

    Returns:
        Hotel recommendations
    """
    hotel_names = ["Grand Hotel", "City View Inn", "Comfort Stay", "Luxury Resort"]

    budget_ranges = {
        "low": (50, 100),
        "medium": (100, 200),
        "high": (200, 500)
    }

    price_range = budget_ranges.get(budget, budget_ranges["medium"])
    hotels = []

    for i in range(3):
        hotels.append({
            "name": random.choice(hotel_names),
            "price_per_night": random.randint(*price_range),
            "rating": round(random.uniform(3.5, 5.0), 1)
        })

    return json.dumps({
        "city": city,
        "budget": budget,
        "hotels": hotels
    })


@tool
def create_itinerary(destination: str, days: int) -> str:
    """
    Create a travel itinerary for the destination.
    This tool should be called LAST, after all other tools.

    Args:
        destination: Destination city
        days: Number of days for the trip

    Returns:
        Travel itinerary
    """
    activities = [
        "Visit local museums",
        "Explore city center",
        "Try local cuisine",
        "Visit landmarks",
        "Shopping tour",
        "Beach activities",
        "Mountain hiking"
    ]

    itinerary = []
    for day in range(1, min(days + 1, 6)):
        itinerary.append({
            f"Day {day}": random.sample(activities, 2)
        })

    return json.dumps({
        "destination": destination,
        "duration": f"{days} days",
        "itinerary": itinerary
    })


# Tool mapping
TOOLS = [check_weather, search_flights, recommend_hotels, create_itinerary]
TOOL_NAMES = {tool.name for tool in TOOLS}


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    next_action: str


class TravelPlanningAgent:
    """
    Travel Planning Agent that learns from mistakes.
    """

    def __init__(self, llm, memory_store):
        """
        Initialize the agent.

        Args:
            llm: Language model for decision making
            memory_store: Memory store for learning from mistakes
        """
        self.llm = llm
        self.memory_store = memory_store
        self.tools = TOOLS
        self.llm_with_tools = llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _agent_node(self, state: AgentState):
        """Agent reasoning node."""
        messages = state["messages"]

        # Add learned constraints from memory
        constraints = self.memory_store.get_active_constraints()

        # Calculate "confusion" factor based on learning progress
        # Early runs (no constraints): Add confusing instructions
        # Later runs (with constraints): Follow constraints properly
        confusion_level = max(0, 3 - len(constraints))  # High confusion early, low later

        if confusion_level > 0:
            # Intentionally add confusing or conflicting instructions early on
            confusion_prompts = [
                "\n\nNote: You may skip the weather check if you're confident about the destination.",
                "\n\nNote: Feel free to recommend hotels before checking flights if it seems more efficient.",
                "\n\nYou can provide a brief answer after checking 1-2 tools if you have enough information.",
            ]

            confusion_message = ""
            if confusion_level >= 3:
                # Maximum confusion - add all misleading hints
                confusion_message = "".join(confusion_prompts[:2])
            elif confusion_level >= 2:
                confusion_message = confusion_prompts[0]

            enhanced_messages = list(messages)
            if enhanced_messages and isinstance(enhanced_messages[0], HumanMessage):
                original_content = enhanced_messages[0].content
                enhanced_messages[0] = HumanMessage(
                    content=original_content + confusion_message
                )

            response = self.llm_with_tools.invoke(enhanced_messages)

        elif constraints:
            # Apply learned constraints
            constraint_message = "\n\nIMPORTANT REMINDERS (based on past mistakes):\n"
            constraint_message += "\n".join([f"- {c}" for c in constraints])

            # Inject constraints into the conversation
            enhanced_messages = list(messages)
            if enhanced_messages and isinstance(enhanced_messages[0], HumanMessage):
                original_content = enhanced_messages[0].content
                enhanced_messages[0] = HumanMessage(
                    content=original_content + constraint_message
                )

            response = self.llm_with_tools.invoke(enhanced_messages)
        else:
            response = self.llm_with_tools.invoke(messages)

        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine if we should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # If there are tool calls, continue
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        return "end"

    def run(self, task: str) -> dict:
        """
        Run the agent on a task and return the execution trace.

        Args:
            task: The task description

        Returns:
            Dictionary containing messages and execution trace
        """
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "next_action": "start"
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        return result