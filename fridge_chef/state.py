"""The shared state that flows between the graph's nodes."""

from typing import Literal, Optional, TypedDict


class ChefState(TypedDict, total=False):
    # Input
    raw_input: str                 # what the user typed
    preference: str                # optional: "בשרי" / "חלבי" / "פרווה" / ""

    # Filled by the graph
    ingredients: list[str]         # cleaned ingredient list
    is_food_request: bool
    recipe: Optional[dict]         # Recipe.model_dump()
    check_errors: list[str]        # problems found by the automatic check
    attempts: int                  # how many recipes were generated so far
    human_feedback: str            # what the user asked to change
    decision: Literal["approve", "revise", "reject", ""]
    shopping_list: list[str]
    final_message: str
