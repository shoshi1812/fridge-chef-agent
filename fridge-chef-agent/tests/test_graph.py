"""Tests the graph flow with a fake LLM (no API key needed)."""

import uuid

from langgraph.types import Command

from fridge_chef import build_graph
from fridge_chef.nodes import find_recipe_problems
from fridge_chef.schemas import ParsedIngredients, Recipe


class FakeLLM:
    """Returns pre-made answers in order, one per call."""

    def __init__(self, answers):
        self.answers = list(answers)

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self.answers.pop(0)


def ing(name, category="parve", available=True):
    return {"name": name, "amount": "1", "category": category, "available": available}


GOOD = Recipe(
    title="שקשוקה",
    kind="parve",
    time_minutes=20,
    ingredients=[ing("ביצה"), ing("עגבנייה"), ing("בצל", available=False)],
    steps=["מטגנים בצל", "מוסיפים עגבניות", "שוברים ביצים"],
)
MIXED = Recipe(
    title="המבורגר עם גבינה",
    kind="meat",
    time_minutes=20,
    ingredients=[ing("בשר טחון", "meat"), ing("גבינה צהובה", "dairy")],
    steps=["מכינים קציצה", "מוסיפים גבינה"],
)
PARSED = ParsedIngredients(is_food_request=True, ingredients=["ביצה", "עגבנייה"])


def run(llm, text="ביצים, עגבניות"):
    graph = build_graph(llm=llm)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke({"raw_input": text, "preference": ""}, config)
    return graph, config, result


def test_not_food():
    _, _, result = run(FakeLLM([ParsedIngredients(is_food_request=False)]), "מה השעה?")
    assert "לא זיהיתי" in result["final_message"]


def test_pauses_for_human_then_approves():
    graph, config, result = run(FakeLLM([PARSED, GOOD]))
    assert "__interrupt__" in result                       # paused at human_review
    result = graph.invoke(Command(resume={"action": "approve"}), config)
    assert result["shopping_list"] == ["בצל"]


def test_retries_when_meat_and_dairy_mixed():
    graph, config, result = run(FakeLLM([PARSED, MIXED, GOOD]))
    assert result["attempts"] == 2                         # second try passed the check
    assert result["recipe"]["title"] == "שקשוקה"


def test_human_revise_loops_back():
    other = GOOD.model_copy(update={"title": "חביתה"})
    graph, config, _ = run(FakeLLM([PARSED, GOOD, other]))
    result = graph.invoke(Command(resume={"action": "revise", "feedback": "משהו מהיר"}), config)
    assert result["recipe"]["title"] == "חביתה"
    result = graph.invoke(Command(resume={"action": "reject"}), config)
    assert "ויתרנו" in result["final_message"]


def test_problem_finder():
    assert find_recipe_problems(MIXED.model_dump()) == ["המתכון מערב בשר וחלב"]
    assert find_recipe_problems(GOOD.model_dump(), "פרווה") == []
    assert find_recipe_problems(GOOD.model_dump(), "חלבי") != []
    assert find_recipe_problems(GOOD.model_dump()) == []
