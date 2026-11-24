"""Runs the agent on every example in the LangSmith dataset and scores it.

Run with:  python -m evaluation.run_eval
Results appear in LangSmith -> Datasets & Experiments -> fridge-chef-examples.
"""

import uuid

from dotenv import load_dotenv
from langgraph.types import Command
from langsmith import Client

from evaluation.dataset import DATASET_NAME, as_langsmith_examples
from fridge_chef import build_graph
from fridge_chef.nodes import PREFERENCE_TO_KIND, find_recipe_problems, missing_ingredients

load_dotenv()
client = Client()
graph = build_graph()


# --------------------------------------------------------------------------- #
# 1. Create the dataset once (skipped if it already exists)
# --------------------------------------------------------------------------- #
def ensure_dataset():
    if client.has_dataset(dataset_name=DATASET_NAME):
        return
    dataset = client.create_dataset(DATASET_NAME, description="Fridge lists for the Fridge Chef agent")
    client.create_examples(dataset_id=dataset.id, examples=as_langsmith_examples())
    print(f"Created dataset '{DATASET_NAME}'")


# --------------------------------------------------------------------------- #
# 2. The target: run the whole graph, auto-approving at the human step
# --------------------------------------------------------------------------- #
def target(inputs: dict) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke(inputs, config)
    if "__interrupt__" in result:
        result = graph.invoke(Command(resume={"action": "approve"}), config)
    return {
        "is_food_request": result.get("is_food_request", False),
        "recipe": result.get("recipe"),
        "attempts": result.get("attempts", 0),
        "preference": inputs.get("preference", ""),
    }


# --------------------------------------------------------------------------- #
# 3. Evaluators: each one returns a score for one run
# --------------------------------------------------------------------------- #
def detects_food_correctly(outputs: dict, reference_outputs: dict) -> bool:
    """Did the agent correctly decide whether the input is a food list?"""
    return outputs["is_food_request"] == reference_outputs["is_food_request"]


def kosher(outputs: dict) -> dict:
    """No meat and dairy together."""
    if not outputs["recipe"]:
        return {"key": "kosher", "score": None, "comment": "no recipe"}
    cats = {i["category"] for i in outputs["recipe"]["ingredients"]}
    return {"key": "kosher", "score": not {"meat", "dairy"} <= cats}


def matches_preference(outputs: dict) -> dict:
    """If the user asked for meat/dairy/parve, did the recipe respect it?"""
    wanted = PREFERENCE_TO_KIND.get(outputs["preference"])
    if not outputs["recipe"] or not wanted:
        return {"key": "matches_preference", "score": None}
    return {"key": "matches_preference", "score": outputs["recipe"]["kind"] == wanted}


def few_missing_ingredients(outputs: dict) -> dict:
    """At most 3 ingredients to buy."""
    if not outputs["recipe"]:
        return {"key": "few_missing_ingredients", "score": None}
    missing = missing_ingredients(outputs["recipe"])
    return {"key": "few_missing_ingredients", "score": len(missing) <= 3, "comment": ", ".join(missing)}


def passed_check_first_try(outputs: dict) -> dict:
    """1 if the first recipe already passed the automatic check, else 0."""
    if not outputs["recipe"]:
        return {"key": "passed_check_first_try", "score": None}
    return {"key": "passed_check_first_try", "score": outputs["attempts"] == 1}


def no_problems(outputs: dict) -> dict:
    """The final recipe has no problems at all (same check the graph uses)."""
    if not outputs["recipe"]:
        return {"key": "no_problems", "score": None}
    problems = find_recipe_problems(outputs["recipe"], outputs["preference"])
    return {"key": "no_problems", "score": not problems, "comment": "; ".join(problems)}


if __name__ == "__main__":
    ensure_dataset()
    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            detects_food_correctly,
            kosher,
            matches_preference,
            few_missing_ingredients,
            passed_check_first_try,
            no_problems,
        ],
        experiment_prefix="fridge-chef",
        max_concurrency=1,          # the free Gemini tier has a low rate limit
    )
    print("Done! Open LangSmith to see the results:", results.experiment_name)
