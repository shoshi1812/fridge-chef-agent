"""Builds the LangGraph StateGraph.

    START -> parse_ingredients --(not food)--> not_food -> END
                     |
                  (food)
                     v
              suggest_recipe <-----------------------------+
                     v                                     |
               check_recipe --(problems & attempts < 3)----+
                     |                                     |
                   (ok)                                    |
                     v                                     |
               human_review --(revise)---------------------+
                |         \\
           (approve)    (reject)
                v           v
      make_shopping_list   rejected -> END
                v
               END
"""

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import ChefState


def route_after_parse(state: ChefState) -> str:
    return "suggest_recipe" if state["is_food_request"] else "not_food"


def route_after_check(state: ChefState) -> str:
    if state["check_errors"] and state["attempts"] < nodes.MAX_ATTEMPTS:
        return "suggest_recipe"          # try again automatically
    return "human_review"                # good enough (or out of retries)


def route_after_human(state: ChefState) -> str:
    return {
        "approve": "make_shopping_list",
        "revise": "suggest_recipe",
    }.get(state["decision"], "rejected")


def build_graph(llm=None, checkpointer=None):
    if llm is None:
        from .llm import get_llm

        llm = get_llm()

    builder = StateGraph(ChefState)

    builder.add_node("parse_ingredients", partial(nodes.parse_ingredients, llm=llm))
    builder.add_node("not_food", nodes.not_food)
    builder.add_node("suggest_recipe", partial(nodes.suggest_recipe, llm=llm))
    builder.add_node("check_recipe", nodes.check_recipe)
    builder.add_node("human_review", nodes.human_review)
    builder.add_node("make_shopping_list", nodes.make_shopping_list)
    builder.add_node("rejected", nodes.rejected)

    builder.add_edge(START, "parse_ingredients")
    builder.add_conditional_edges("parse_ingredients", route_after_parse, ["suggest_recipe", "not_food"])
    builder.add_edge("suggest_recipe", "check_recipe")
    builder.add_conditional_edges("check_recipe", route_after_check, ["suggest_recipe", "human_review"])
    builder.add_conditional_edges(
        "human_review", route_after_human, ["make_shopping_list", "suggest_recipe", "rejected"]
    )
    builder.add_edge("not_food", END)
    builder.add_edge("make_shopping_list", END)
    builder.add_edge("rejected", END)

    # The checkpointer saves the state after every step. It is what lets the
    # graph pause at interrupt() and continue later from the same point.
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
