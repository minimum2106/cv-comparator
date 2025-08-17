from typing import List, Dict, Any, Type
from dotenv import load_dotenv
from pathlib import Path
import tomllib
import json
import os

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

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


class ComparatorInput(BaseModel):
    cv_data: List[str] = Field(
        description="CV's file content, emphasizing the details of each candidate"
    )
    evaluation_criteria: Dict[str, str] = Field(
        description="Evaluation criteria with either 'must have' or 'nice to have' (e.g., {'python': 'must have', 'experience': 'nice to have', 'education': 'must have', 'projects': 'nice to have'})"
    )


class CriteriaAnalysis(BaseModel):
    """Structured output for criteria analysis"""

    criteria_found: List[str] = Field(
        description="List of criteria that are present in the CV"
    )
    criteria_missing: List[str] = Field(
        description="List of criteria that are missing from the CV"
    )
    analysis_notes: str = Field(
        description="Brief notes about the candidate's profile and strengths"
    )


class ComparatorAgent(BaseTool):

    name: str = "cv_comparator"
    description: str = (
        "This agent compares CVs against evaluation criteria with must-have and nice-to-have priorities using LLM analysis"
    )
    args_schema: Type[BaseModel] = ComparatorInput

    def __init__(self):
        super().__init__()

    def _analyze_cv_with_llm(
        self, cv_content: str, criteria_list: List[str], candidate_name: str
    ) -> CriteriaAnalysis:
        """Use LLM to analyze CV against criteria"""

        criteria_text = "\n".join([f"- {criterion}" for criterion in criteria_list])

        analysis_prompt = f"""
        Analyze the following CV against the specified criteria and determine which criteria the candidate meets.
        
        CANDIDATE: {candidate_name}
        
        CV CONTENT:
        {cv_content}
        
        CRITERIA TO EVALUATE:
        {criteria_text}
        
        INSTRUCTIONS:
        1. First, translate the CV content to English if it is in another language.
        2. Identify which criteria from the list are present in the CV.
        3. Carefully analyze the CV content for evidence of each criterion
        4. Consider both explicit mentions and implicit evidence (e.g., job titles, projects, experience descriptions)
        5. Be thorough but not overly strict - look for reasonable evidence
        6. For technical skills, look for mentions in skills sections, job descriptions, or project descriptions
        7. For experience/education, look at work history and educational background
        8. For soft skills, look at leadership roles, team descriptions, achievements

        EXAMPLES OF EVIDENCE:
        - "Python": Mentioned in skills, used in projects, or mentioned in job descriptions
        - "Leadership": Team lead roles, managed teams, led projects, mentoring experience
        - "Experience": Years of work experience, relevant job positions
        - "AWS": Cloud experience, specific AWS services mentioned, cloud projects
        - "Education": Degrees, certifications, relevant coursework
        
        Return only the criteria that have clear evidence in the CV. Do not include criteria if the evidence is weak or unclear.
        """

        try:
            structured_llm = LLM.with_structured_output(CriteriaAnalysis)
            response = structured_llm.invoke([HumanMessage(content=analysis_prompt)])

            return response

        except Exception as e:
            print(f"❌ LLM analysis failed for {candidate_name}: {e}")
            # Fallback to simple keyword matching
            criteria_found = []
            criteria_missing = []

            cv_lower = cv_content.lower()
            for criterion in criteria_list:
                if criterion.lower() in cv_lower:
                    criteria_found.append(criterion)
                else:
                    criteria_missing.append(criterion)

            return CriteriaAnalysis(
                criteria_found=criteria_found,
                criteria_missing=criteria_missing,
                analysis_notes=f"Fallback analysis for {candidate_name} (LLM unavailable)",
            )

    def _calculate_max_possible_score(
        self, evaluation_criteria: Dict[str, str]
    ) -> float:
        """Calculate maximum possible score: len(must_have) + len(nice_to_have) * 0.5"""
        must_have_count = sum(
            1
            for priority in evaluation_criteria.values()
            if priority.lower() == "must have"
        )
        nice_to_have_count = sum(
            1
            for priority in evaluation_criteria.values()
            if priority.lower() == "nice to have"
        )

        return must_have_count + (nice_to_have_count * 0.5)

    def _run(self, cv_data: List[str], evaluation_criteria: Dict[str, str]) -> str:
        """
        Load CVs from directory and apply LLM-based criteria analysis
        """

        # Separate criteria by priority
        must_have_criteria = [
            k for k, v in evaluation_criteria.items() if v.lower() == "must have"
        ]
        nice_to_have_criteria = [
            k for k, v in evaluation_criteria.items() if v.lower() == "nice to have"
        ]
        all_criteria = list(evaluation_criteria.keys())

        # Calculate maximum possible score
        max_possible_score = len(must_have_criteria) + (
            len(nice_to_have_criteria) * 0.5
        )
        print(f"   📈 Maximum Possible Score: {max_possible_score}")

        results = []

        for i, cv in enumerate(cv_data):
            candidate_name = cv.get("name", f"Candidate_{i+1}")
            cv_content = cv.get("content", "")

            print(f"🔍 Analyzing: {candidate_name} with LLM...")

            # Use LLM to analyze CV against criteria
            analysis = self._analyze_cv_with_llm(
                cv_content, all_criteria, candidate_name
            )

            # Calculate scores based on LLM analysis
            raw_score = 0
            criterion_results = {}

            # Check must-have criteria (+1 point each)
            for criterion in must_have_criteria:
                if criterion in analysis.criteria_found:
                    raw_score += 1

            # Check nice-to-have criteria (+0.5 points each)
            for criterion in nice_to_have_criteria:
                if criterion in analysis.criteria_found:
                    raw_score += 0.5

            # Calculate percentage score
            percentage = (
                round((raw_score / max_possible_score) * 100, 2)
                if max_possible_score > 0
                else 0
            )

            # Create candidate result
            candidate_result = {
                "name": candidate_name,
                "filename": cv.get("filename"),
                "raw_score": raw_score,
                "max_possible_score": max_possible_score,
                "percentage": percentage,
                "criterion_results": criterion_results,
                "llm_analysis": {
                    "criteria_found": analysis.criteria_found,
                    "criteria_missing": analysis.criteria_missing,
                    "analysis_notes": analysis.analysis_notes,
                },
            }

            results.append(candidate_result)

        # Sort by raw score (highest first)
        results.sort(key=lambda x: (x["raw_score"]), reverse=True)

        # Generate comparison table
        comparison_table = self._generate_comparison_table(results, evaluation_criteria)

        return json.dumps(
            {
                "total_candidates": len(results),
                "analysis_method": "LLM-based",
                "scoring_system": {
                    "must_have_points": 1.0,
                    "nice_to_have_points": 0.5,
                    "max_possible_score": max_possible_score,
                    "formula": "score = (must_have_count * 1.0) + (nice_to_have_count * 0.5)",
                },
                "evaluation_summary": {
                    "must_have_criteria": must_have_criteria,
                    "nice_to_have_criteria": nice_to_have_criteria,
                },
                "ranked_candidates": results,
                "comparison_table": comparison_table,
            },
            indent=2,
        )

    def _generate_comparison_table(
        self, results: List[Dict], criteria: Dict[str, str]
    ) -> str:
        """Generate a simplified markdown comparison table"""

        # Separate criteria by priority for header
        must_have_criteria = [
            k for k, v in criteria.items() if v.lower() == "must have"
        ]
        nice_to_have_criteria = [
            k for k, v in criteria.items() if v.lower() == "nice to have"
        ]

        # Table header
        header = "| Rank | Candidate | Filename | "

        # Add must-have columns
        for criterion in must_have_criteria:
            header += f"{criterion} (Must) | "

        # Add nice-to-have columns
        for criterion in nice_to_have_criteria:
            header += f"{criterion} (Nice) | "

        header += "Raw Score | Max Score | Percentage | LLM Notes |\n"

        # Separator
        total_columns = (
            3 + len(criteria) + 4
        )  # Rank, Candidate, Filename + criteria + scores + notes
        separator = "|" + "|".join(["------"] * total_columns) + "|\n"

        # Table rows
        rows = []
        for rank, candidate in enumerate(results, 1):
            row = f"| {rank} | {candidate['name']} | {candidate.get('filename', 'N/A')} | "

            # Add must-have results
            for criterion in must_have_criteria:
                result = candidate["criterion_results"].get(
                    f"{criterion} (Must)", "❌ Missing"
                )
                status = "✅" if "Found" in result else "❌"
                row += f"{status} | "

            # Add nice-to-have results
            for criterion in nice_to_have_criteria:
                result = candidate["criterion_results"].get(
                    f"{criterion} (Nice)", "❌ Missing"
                )
                status = "✅" if "Found" in result else "❌"
                row += f"{status} | "

            # Add summary scores
            row += f"{candidate['raw_score']} | "
            row += f"{candidate['max_possible_score']} | "
            row += f"{candidate['percentage']}% | "

            # Add LLM analysis notes (truncated)
            notes = candidate.get("llm_analysis", {}).get("analysis_notes", "No notes")
            truncated_notes = notes[:50] + "..." if len(notes) > 50 else notes
            row += f"{truncated_notes} |"

            rows.append(row)

        return header + separator + "\n".join(rows)

    def _load_cv_data(self, cv_directory: str) -> List[Dict[str, Any]]:
        """Load CV data from the specified directory"""
        cv_data = []
        if not os.path.exists(cv_directory):
            raise ValueError(f"CV directory '{cv_directory}' does not exist")

        for filename in os.listdir(cv_directory):
            if filename.endswith(".txt"):
                file_path = Path(cv_directory) / filename
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:  # Only include non-empty files
                            cv_data.append(
                                {
                                    "name": filename,
                                    "content": content,
                                    "filename": filename,
                                }
                            )
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        return cv_data
