"""Structured outputs the LLM must return (validated with Pydantic)."""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["meat", "dairy", "parve"]


class ParsedIngredients(BaseModel):
    """Result of reading the user's free-text list of what is in the fridge."""

    is_food_request: bool = Field(
        description="True if the text is a list of food ingredients (in any language)."
    )
    ingredients: list[str] = Field(
        default_factory=list,
        description="Clean, de-duplicated ingredient names in Hebrew, singular form.",
    )


class RecipeIngredient(BaseModel):
    name: str = Field(description="Ingredient name in Hebrew.")
    amount: str = Field(description="Amount, e.g. '2 כוסות' or '3 יחידות'.")
    category: Category = Field(
        description="Kashrut category of this ingredient: meat, dairy or parve."
    )
    available: bool = Field(
        description="True if this ingredient is in the user's fridge list."
    )


class Recipe(BaseModel):
    title: str = Field(description="Recipe name in Hebrew.")
    kind: Category = Field(description="Kashrut type of the whole dish.")
    time_minutes: int = Field(description="Total preparation time in minutes.")
    ingredients: list[RecipeIngredient]
    steps: list[str] = Field(description="Preparation steps in Hebrew, in order.")
