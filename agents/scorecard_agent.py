import tomllib
import json

from langchain_core.tools import BaseTool
from typing import List, Type, Dict, Any
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agents.prompts import SCORECARD_AGENT_PROMPT


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


class ScorecardInput(BaseModel):
    job_brief: str = Field(
        description="""
            HR job brief containing job description and requirements.
        """
    )


class CriteriaExtraction(BaseModel):
    """Structured output for criteria extraction"""

    must_have: List[str] = Field(
        description="Essential criteria that candidates must possess"
    )
    nice_to_have: List[str] = Field(
        description="Preferred criteria that would be beneficial but not essential"
    )


class ScorecardAgent(BaseTool):
    """A simple agent for generating evaluation scorecards from job briefs using LLM."""

    name: str = "scorecard_generator"
    description: str = (
        """Generate evaluation criteria scorecard from job brief text. 
        Extracts requirements and categorizes them into must-have and nice-to-have criteria.
        """
    )
    args_schema: Type[BaseModel] = ScorecardInput

    def __init__(self):
        super().__init__()

    def _extract_criteria_with_llm(self, job_brief: str) -> dict:
        """Use LLM to extract must-have and nice-to-have criteria from job brief"""

        extraction_prompt = SCORECARD_AGENT_PROMPT.format(job_brief=job_brief)

        try:
            structured_llm = LLM.with_structured_output(CriteriaExtraction)
            response = structured_llm.invoke([HumanMessage(content=extraction_prompt)])

            print(
                f"✅ Extracted {len(response.must_have)} must-have and {len(response.nice_to_have)} nice-to-have criteria"
            )

            return response.model_dump()

        except Exception as e:
            print(f"❌ LLM criteria extraction failed: {e}")
            # Fallback to default criteria
            return CriteriaExtraction(
                must_have=[],
                nice_to_have=[],
            ).model_dump()

    def _run(self, job_brief: str) -> dict:
        """
        Create evaluation scorecard from job brief using LLM
        """
        print(f"🧠 Using LLM to analyze job brief...")

        # Extract criteria using LLM
        criteria_extraction = self._extract_criteria_with_llm(job_brief)

        return criteria_extraction
