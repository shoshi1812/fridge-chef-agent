"""Streamlit UI for the Fridge Chef agent.

Run with:  streamlit run app.py
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from fridge_chef import build_graph

load_dotenv()

STEP_NAMES = {
    "parse_ingredients": "🧾 קורא את רשימת המצרכים",
    "not_food": "🤔 זה לא נראה כמו רשימת מצרכים",
    "suggest_recipe": "👨‍🍳 מחפש מתכון",
    "check_recipe": "✅ בודק כשרות ומצרכים חסרים",
    "human_review": "🙋 ממתין לאישור שלך",
    "make_shopping_list": "🛒 מכין רשימת קניות",
    "rejected": "↩️ ויתרנו על המתכון",
}

st.set_page_config(page_title="מה יש במקרר?", page_icon="🍳")
st.markdown(
    "<style>body, .stApp, textarea, input {direction: rtl; text-align: right;}</style>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Session state: one graph + one thread_id per browser session
# --------------------------------------------------------------------------- #
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

graph = st.session_state.graph
config = {
    "configurable": {"thread_id": st.session_state.thread_id},
    "run_name": "fridge-chef",                 # name of the run in LangSmith
    "metadata": {"ui": "streamlit"},
}


def run_graph(graph_input):
    """Run the graph until it ends or pauses, showing each step as it happens."""
    with st.status("הסוכן עובד...", expanded=True) as status:
        for update in graph.stream(graph_input, config, stream_mode="updates"):
            for node in update:
                if node in STEP_NAMES:
                    st.write(STEP_NAMES[node])
        status.update(label="סיום", state="complete", expanded=False)
    st.rerun()


def show_recipe(recipe):
    kinds = {"meat": "בשרי", "dairy": "חלבי", "parve": "פרווה"}
    st.subheader(recipe["title"])
    st.caption(f"{kinds[recipe['kind']]} · {recipe['time_minutes']} דקות")
    st.markdown("**מצרכים:**")
    for ing in recipe["ingredients"]:
        mark = "✅" if ing["available"] else "🛒"
        st.markdown(f"- {mark} {ing['name']} ({ing['amount']})")
    st.markdown("**אופן ההכנה:**")
    for i, step in enumerate(recipe["steps"], 1):
        st.markdown(f"{i}. {step}")


def restart():
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("איך זה עובד?")
    st.write(
        "כותבים מה יש במקרר, והסוכן (LangGraph + Gemini) מציע מתכון כשר, "
        "בודק אותו, מחכה לאישור שלך ומכין רשימת קניות."
    )
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        st.success(f"LangSmith tracing פעיל · פרויקט: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    else:
        st.warning("LangSmith tracing כבוי (בדקו את קובץ ה-.env)")
    if st.button("התחלה מחדש"):
        restart()


# --------------------------------------------------------------------------- #
# Main screen: what we show depends on where the graph is
# --------------------------------------------------------------------------- #
st.title("🍳 מה יש במקרר?")
snapshot = graph.get_state(config)
values = snapshot.values

if not values:
    # Nothing started yet: ask for ingredients
    text = st.text_area("מה יש לך במקרר?", placeholder="למשל: ביצים, עגבניות, בצל, גבינה לבנה, פסטה")
    preference = st.radio("איזו מנה?", ["לא משנה", "בשרי", "חלבי", "פרווה"], horizontal=True)
    if st.button("מצא לי מתכון", type="primary", disabled=not text.strip()):
        run_graph({"raw_input": text, "preference": "" if preference == "לא משנה" else preference})

elif snapshot.interrupts:
    # Graph is paused at human_review and waiting for us
    payload = snapshot.interrupts[0].value
    show_recipe(payload["recipe"])
    if payload["warnings"]:
        st.warning("שימו לב: " + "; ".join(payload["warnings"]))

    st.divider()
    st.markdown(f"**{payload['question']}**")
    col1, col2 = st.columns(2)
    if col1.button("👍 מאשרת", type="primary"):
        run_graph(Command(resume={"action": "approve"}))
    if col2.button("👎 לא בא לי"):
        run_graph(Command(resume={"action": "reject"}))
    feedback = st.text_input("או בקשו שינוי:", placeholder="למשל: בלי בצל, משהו יותר מהיר")
    if st.button("🔄 שנה את המתכון", disabled=not feedback.strip()):
        run_graph(Command(resume={"action": "revise", "feedback": feedback}))

else:
    # Graph reached END
    st.info(values.get("final_message", ""))
    if values.get("decision") == "approve":
        show_recipe(values["recipe"])
        st.subheader("🛒 רשימת קניות")
        if values["shopping_list"]:
            for item in values["shopping_list"]:
                st.checkbox(item, key=f"buy-{item}")
        else:
            st.write("יש לך הכל בבית!")
    if st.button("מתכון חדש", type="primary"):
        restart()
