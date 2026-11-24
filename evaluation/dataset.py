"""Test examples for the LangSmith evaluation.

Each example: the input the user types + what we expect.
"""

DATASET_NAME = "fridge-chef-examples"

EXAMPLES = [
    # (raw_input, preference, is_food)
    ("ביצים, עגבניות, בצל, פלפל אדום", "", True),
    ("פסטה, גבינה צהובה, שמנת מתוקה, פטריות", "חלבי", True),
    ("חזה עוף, תפוחי אדמה, גזר, שום", "בשרי", True),
    ("אורז, עדשים, בצל, כמון", "פרווה", True),
    ("בשר טחון, גבינה, לחמניות, עגבנייה", "", True),          # tempting to mix meat & dairy!
    ("קמח, ביצים, חלב, סוכר, תפוחים", "חלבי", True),
    ("טונה, מלפפון, תירס, מיונז, לחם", "", True),
    ("שניצל, פתיתים, אפונה", "בשרי", True),
    ("קוטג', לחם, עגבנייה, זיתים", "פרווה", True),          # asks for parve but has dairy
    ("מה מזג האוויר מחר?", "", False),
    ("תכתוב לי שיר על ים", "", False),
]


def as_langsmith_examples():
    return [
        {
            "inputs": {"raw_input": text, "preference": pref},
            "outputs": {"is_food_request": is_food},
        }
        for text, pref, is_food in EXAMPLES
    ]
