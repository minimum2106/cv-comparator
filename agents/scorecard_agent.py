from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Type, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import json


class ScorecardInput(BaseModel):
    job_brief: str = Field(
        description="HR job brief describing the position requirements"
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
        "Analyze and Generate evaluation scorecard from a HR job brief using LLM analysis"
    )
    args_schema: Type[BaseModel] = ScorecardInput

    def __init__(self):
        super().__init__()

    def _extract_criteria_with_llm(self, job_brief: str) -> CriteriaExtraction:
        """Use LLM to extract must-have and nice-to-have criteria from job brief"""

        extraction_prompt = f"""
        Analyze the following job brief and extract evaluation criteria for candidate assessment.
        
        JOB BRIEF:
        {job_brief}
        
        INSTRUCTIONS:
        1. Identify MUST-HAVE criteria - Essential qualifications, skills, and experience that are absolutely required for the role
        2. Identify NICE-TO-HAVE criteria - Preferred qualifications, skills, and experience that would be beneficial but not essential
        3. Focus on specific, measurable criteria that can be evaluated in a candidate
        4. Include both technical skills and soft skills where applicable
        5. Avoid overly broad or vague criteria
        6. Each criterion should be 2-8 words long and clearly defined
        
        EXAMPLES OF GOOD CRITERIA:
        - "3+ years Python experience"
        - "Bachelor's degree in Computer Science"
        - "REST API development experience"
        - "Team leadership skills"
        - "Agile methodology experience"
        
        Extract 5-8 must-have criteria and 3-6 nice-to-have criteria.
        """

        try:
            # Use structured output to get criteria
            llm = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.0,
                streaming=False,
            )

            structured_llm = llm.with_structured_output(CriteriaExtraction)
            response = structured_llm.invoke([HumanMessage(content=extraction_prompt)])

            print(
                f"✅ Extracted {len(response.must_have)} must-have and {len(response.nice_to_have)} nice-to-have criteria"
            )

            return response

        except Exception as e:
            print(f"❌ LLM criteria extraction failed: {e}")
            # Fallback to default criteria
            return CriteriaExtraction(
                must_have=[
                    "Relevant technical experience",
                    "Required programming skills",
                    "Educational qualifications",
                    "Communication abilities",
                    "Problem-solving skills",
                ],
                nice_to_have=[
                    "Additional certifications",
                    "Leadership experience",
                    "Industry knowledge",
                ],
            )

    def _extract_position_title_with_llm(self, job_brief: str) -> str:
        """Use LLM to extract position title from job brief"""

        title_prompt = f"""
        Extract the job position title from the following job brief.
        
        JOB BRIEF:
        {job_brief}
        
        INSTRUCTIONS:
        - Return only the job title/position name
        - Make it concise and professional
        - If no explicit title is mentioned, infer it from the job description
        - Examples: "Senior Software Engineer", "Product Manager", "Data Analyst"
        
        Position Title:
        """

        try:
            response = self.llm.invoke([HumanMessage(content=title_prompt)])
            title = response.content.strip().strip("\"'")

            if title and len(title) < 100:  # Reasonable title length
                return title
            else:
                return "Position Not Specified"

        except Exception as e:
            print(f"❌ Title extraction failed: {e}")
            return "Position Not Specified"

    def _run(self, job_brief: str) -> str:
        """
        Create evaluation scorecard from job brief using LLM
        """
        print(f"🧠 Using LLM to analyze job brief...")

        # Extract position title using LLM
        position_title = self._extract_position_title_with_llm(job_brief)
        print(f"📋 Position: {position_title}")

        # Extract criteria using LLM
        criteria_extraction = self._extract_criteria_with_llm(job_brief)

        # Generate scorecard
        scorecard = {
            "position": position_title,
            "job_brief_analyzed": True,
            "extraction_method": "LLM-based",
            "must_have_criteria": criteria_extraction.must_have,
            "nice_to_have_criteria": criteria_extraction.nice_to_have,
            "total_criteria": len(criteria_extraction.must_have)
            + len(criteria_extraction.nice_to_have),
            "scorecard_format": self._generate_scorecard_format(criteria_extraction),
            "evaluation_guide": self._generate_evaluation_guide(criteria_extraction),
            "criteria_breakdown": {
                "must_have_count": len(criteria_extraction.must_have),
                "nice_to_have_count": len(criteria_extraction.nice_to_have),
                "must_have_weight": 70,
                "nice_to_have_weight": 30,
            },
        }

        return json.dumps(scorecard, indent=2)

    def _generate_scorecard_format(self, criteria: CriteriaExtraction) -> str:
        """Generate scorecard format as markdown table"""

        scorecard = "## Evaluation Scorecard\n\n"

        # Must-have criteria section (70% total weight)
        scorecard += "### Must-Have Criteria (70% total weight)\n\n"
        scorecard += "| Criterion | Score (1-10) | Weight | Weighted Score | Notes |\n"
        scorecard += "|-----------|--------------|--------|----------------|-------|\n"

        must_have_individual_weight = (
            70 // len(criteria.must_have) if criteria.must_have else 0
        )
        remainder = 70 % len(criteria.must_have) if criteria.must_have else 0

        for i, criterion in enumerate(criteria.must_have):
            # Add remainder to first criterion to ensure total is 70
            weight = must_have_individual_weight + (remainder if i == 0 else 0)
            scorecard += f"| {criterion} | ___ | {weight}% | ___ | ___ |\n"

        # Nice-to-have criteria section (30% total weight)
        if criteria.nice_to_have:
            scorecard += "\n### Nice-to-Have Criteria (30% total weight)\n\n"
            scorecard += (
                "| Criterion | Score (1-10) | Weight | Weighted Score | Notes |\n"
            )
            scorecard += (
                "|-----------|--------------|--------|----------------|-------|\n"
            )

            nice_to_have_individual_weight = 30 // len(criteria.nice_to_have)
            remainder = 30 % len(criteria.nice_to_have)

            for i, criterion in enumerate(criteria.nice_to_have):
                # Add remainder to first criterion to ensure total is 30
                weight = nice_to_have_individual_weight + (remainder if i == 0 else 0)
                scorecard += f"| {criterion} | ___ | {weight}% | ___ | ___ |\n"

        scorecard += "\n**TOTAL SCORE: ___/100**\n"
        scorecard += "\n**RECOMMENDATION: ___**\n"

        return scorecard

    def _generate_evaluation_guide(
        self, criteria: CriteriaExtraction
    ) -> Dict[str, str]:
        """Generate evaluation guide for each criterion"""
        guides = {}

        # Generate guides for must-have criteria
        for criterion in criteria.must_have:
            guides[f"Must-Have: {criterion}"] = (
                f"CRITICAL - Evaluate candidate's {criterion.lower()}. This is essential for role success and should be thoroughly assessed."
            )

        # Generate guides for nice-to-have criteria
        for criterion in criteria.nice_to_have:
            guides[f"Nice-to-Have: {criterion}"] = (
                f"PREFERRED - Assess candidate's {criterion.lower()}. This adds value but is not a disqualifier if absent."
            )

        return guides

    def run(self, input_data: Dict[str, Any]) -> str:
        """Run the scorecard generator with input data"""
        return self._run(job_brief=input_data.get("job_brief", ""))
