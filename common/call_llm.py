
from dotenv import load_dotenv
import tomllib

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()
def get_llm():
    with open("project.toml", "rb") as f:
        config = tomllib.load(f)
        provider = config.get("project", {}).get("models").get("provider")

        if provider == "openai":
            model = config.get("project", {}).get("models").get("openai_default")
            LLM = ChatOpenAI(model=model, temperature=0.0, streaming=False)
        elif provider == "groq":
            model = config.get("project", {}).get("models").get("groq_default")
            LLM = ChatGroq(model=model, temperature=0.0, streaming=False)
        else:
            raise ValueError("Unsupported model provider")

    return LLM
