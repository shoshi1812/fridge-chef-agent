"""The graph's nodes. Each node gets the state and returns only the fields it changes."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from .schemas import ParsedIngredients, Recipe

MAX_ATTEMPTS = 3        # automatic retries before we ask the human anyway
MAX_MISSING = 3         # how many ingredients the user may need to buy

# Basics we assume every kitchen has, so they don't count as "missing".
PANTRY_BASICS = {"מלח", "פלפל", "פלפל שחור", "שמן", "שמן זית", "מים", "סוכר"}

PREFERENCE_TO_KIND = {"בשרי": "meat", "חלבי": "dairy", "פרווה": "parve"}


# --------------------------------------------------------------------------- #
# 1. Read the user's text
# --------------------------------------------------------------------------- #
def parse_ingredients(state, llm):
    parser = llm.with_structured_output(ParsedIngredients)
    result: ParsedIngredients = parser.invoke(
        [
            SystemMessage(
                "You read a short free text that should list food items a person "
                "has at home. Decide whether it is really a list of food ingredients, "
                "and if so return a clean list of ingredient names in Hebrew."
            ),
            HumanMessage(state["raw_input"]),
        ]
    )
    return {
        "is_food_request": result.is_food_request and len(result.ingredients) > 0,
        "ingredients": result.ingredients,
        "attempts": 0,
        "check_errors": [],
        "human_feedback": "",
    }


def not_food(state):
    return {
        "final_message": "לא זיהיתי רשימת מצרכים. נסו לכתוב מה יש לכם במקרר, למשל: ביצים, עגבניות, גבינה."
    }


# --------------------------------------------------------------------------- #
# 2. Ask the model for a recipe
# --------------------------------------------------------------------------- #
def suggest_recipe(state, llm):
    generator = llm.with_structured_output(Recipe)

    instructions = [
        "אתה שף ביתי. הצע מתכון אחד פשוט וטעים בעברית.",
        f"המצרכים שיש בבית: {', '.join(state['ingredients'])}.",
        f"השתמש בעיקר במצרכים האלה. מותר להוסיף לכל היותר {MAX_MISSING} מצרכים שצריך לקנות "
        f"(לא כולל בסיסיים: {', '.join(sorted(PANTRY_BASICS))}).",
        "המתכון חייב להיות כשר: אסור לערב בשר וחלב באותו מתכון.",
        "לכל מצרך ציין קטגוריה (meat/dairy/parve) והאם הוא קיים ברשימה של המשתמש (available).",
    ]
    if state.get("preference"):
        instructions.append(f"המשתמש ביקש מנה {state['preference']}.")
    if state.get("check_errors"):
        instructions.append("בניסיון הקודם נמצאו הבעיות הבאות, תקן אותן: " + "; ".join(state["check_errors"]))
    if state.get("human_feedback"):
        previous = state.get("recipe") or {}
        instructions.append(
            f"המשתמש ראה את המתכון '{previous.get('title', '')}' וביקש שינוי: {state['human_feedback']}"
        )

    recipe: Recipe = generator.invoke([HumanMessage("\n".join(instructions))])
    return {
        "recipe": recipe.model_dump(),
        "attempts": state.get("attempts", 0) + 1,
        "human_feedback": "",
    }


# --------------------------------------------------------------------------- #
# 3. Automatic quality check (plain Python, no LLM)
# --------------------------------------------------------------------------- #
def find_recipe_problems(recipe: dict, preference: str = "") -> list[str]:
    """Return a list of problems in the recipe. Empty list = recipe is OK.

    Also used by the LangSmith evaluation.
    """
    problems = []
    categories = {ing["category"] for ing in recipe["ingredients"]}

    if {"meat", "dairy"} <= categories:
        problems.append("המתכון מערב בשר וחלב")

    wanted = PREFERENCE_TO_KIND.get(preference)
    if wanted and recipe["kind"] != wanted:
        problems.append(f"ביקשו מנה {preference} אבל המתכון לא כזה")
    if wanted == "parve" and categories & {"meat", "dairy"}:
        problems.append("ביקשו פרווה אבל יש מצרכים בשריים או חלביים")
    elif wanted == "dairy" and "meat" in categories:
        problems.append("ביקשו חלבי אבל יש מצרך בשרי")
    elif wanted == "meat" and "dairy" in categories:
        problems.append("ביקשו בשרי אבל יש מצרך חלבי")

    missing = missing_ingredients(recipe)
    if len(missing) > MAX_MISSING:
        problems.append(f"חסרים יותר מדי מצרכים ({len(missing)}): {', '.join(missing)}")

    if len(recipe["steps"]) < 2:
        problems.append("אין מספיק שלבי הכנה")

    return problems


def missing_ingredients(recipe: dict) -> list[str]:
    return [
        ing["name"]
        for ing in recipe["ingredients"]
        if not ing["available"] and ing["name"] not in PANTRY_BASICS
    ]


def check_recipe(state):
    return {"check_errors": find_recipe_problems(state["recipe"], state.get("preference", ""))}


# --------------------------------------------------------------------------- #
# 4. Human in the loop: pause the graph and wait for the user
# --------------------------------------------------------------------------- #
def human_review(state):
    answer = interrupt(
        {
            "recipe": state["recipe"],
            "warnings": state.get("check_errors", []),
            "question": "לאשר את המתכון, לבקש שינוי או לוותר?",
        }
    )
    # `answer` is whatever the UI sends back with Command(resume=...)
    decision = answer.get("action", "reject")
    update = {"decision": decision, "human_feedback": answer.get("feedback", "")}
    if decision == "revise":
        update["attempts"] = 0          # a new round of automatic retries
        update["check_errors"] = []
    return update


# --------------------------------------------------------------------------- #
# 5. Final steps
# --------------------------------------------------------------------------- #
def make_shopping_list(state):
    items = missing_ingredients(state["recipe"])
    return {
        "shopping_list": items,
        "final_message": f"בתיאבון! המתכון '{state['recipe']['title']}' אושר.",
    }


def rejected(state):
    return {"final_message": "בסדר, ויתרנו על המתכון. אפשר להתחיל מחדש עם מצרכים אחרים."}
