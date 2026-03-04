from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_report(state):

    hazard = state["hazard"]

    prompt = f"""
    Generate an environmental incident report.

    Hazard detected: {hazard}

    Explain possible ecological impact and recommended actions.
    """

    response = llm.invoke(prompt)

    return {"report": response.content}