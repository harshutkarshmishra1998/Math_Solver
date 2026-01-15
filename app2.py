import api_keys  # loads GROQ_API_KEY

import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import numexpr as ne
import re


# -------------------- STREAMLIT SETUP --------------------
st.set_page_config(
    page_title="Text to Math Solver",
    page_icon="🧮",
    layout="centered"
)

st.title("🧮 Text → Math Problem Solver")
st.caption("LLM for interpretation, Python for calculation (correct & deterministic)")


# -------------------- LLM --------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    stop_sequences=[]
)


# -------------------- WORD → EXPRESSION --------------------
expression_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Convert the following word problem into ONE valid mathematical expression.

RULES:
- Output ONLY the expression
- No explanation
- No text
- No code
- Use + - * / and parentheses

Question: {question}
Expression:
"""
)

expression_chain = LLMChain(
    llm=llm,
    prompt=expression_prompt
)


# -------------------- SAFE PYTHON CALCULATOR --------------------
def evaluate_expression(expression: str) -> str:
    """
    Deterministically evaluate math using Python.
    No LLM involved.
    """
    # safety: only allow math characters
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        raise ValueError(f"Unsafe expression: {expression}")

    result = ne.evaluate(expression)
    return str(result.item())


# -------------------- UI --------------------
question = st.text_area(
    "Enter your math word problem:",
    value=(
        "I have 5 bananas and 7 grapes. "
        "I eat 2 bananas and give away 3 grapes. "
        "Then I buy a dozen apples and 2 packs of blueberries. "
        "Each pack of blueberries contains 25 berries. "
        "How many total pieces of fruit do I have at the end?"
    ),
    height=160
)

if st.button("Solve"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Solving..."):
            try:
                # 1️⃣ LLM → expression
                expression = expression_chain.run(question).strip()

                # 2️⃣ Python → calculate
                final_answer = evaluate_expression(expression)

                # -------------------- DISPLAY --------------------
                st.subheader("🧠 Interpreted Expression")
                st.code(expression)

                st.subheader("✅ Final Answer")
                st.success(final_answer)

            except Exception as e:
                st.error("Failed to solve the problem.")
                st.exception(e)