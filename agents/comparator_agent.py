from typing import List, Dict, Type
import json
import os

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agents.utils import read_txt_file
from common.call_llm import get_llm


    

class ReadTxtDirectoryInput(BaseModel):
    directory: str = Field(
        description="Extract the directory path containing .txt files to read.",
    )


class FileContent(BaseModel):
    file_name: str = Field(
        description="The name of the .txt file.",
    )
    content: str = Field(
        description="The extracted content of the .txt file.",
    )


class ReadTxtDirectoryOutput(BaseModel):
    file_contents: List[FileContent] = Field(
        default_factory=list,
        description="The combined contents of all .txt files in the directory.",
    )


class ComparatorInput(BaseModel):
    cv_directory: str = Field(
        description="Directory containing CV txt files"
    )
    evaluation_criteria: Dict[str, str] = Field(
        description="""
            Evaluation criteria with either 'must have' or 'nice to have' 
            (e.g., {'Have experience working in startups': 'must have', 
                    'Machine Learning experience': 'nice to have', 
                    'Bachelor degree in Computer Science': 'must have', 
                    'AWS cloud experience': 'nice to have',
                    'Leadership experience': 'nice to have',
                    'French language skills': 'must have',
            })
        """
    )


class CriteriaAnalysis(BaseModel):
    """Structured output for criteria analysis"""

    criteria_found: List[str] = Field(
        default_factory=list,
        description="List of criteria that are present in the CV"
    )
    criteria_missing: List[str] = Field(
        default_factory=list,
        description="List of criteria that are missing from the CV"
    )
    analysis_notes: str = Field(
        default="",
        description="Brief notes about the candidate's profile and strengths"
    )


class ComparatorAgent(BaseTool):

    name: str = "cv_comparator"
    description: str = (
        "Read all txt files in a directory"
        "wrap the content of each file into file_name and content "
        "and compares CVs against evaluation criteria with must-have and nice-to-have priorities using LLM analysis"
        "and returns a structured report"
    )
    args_schema: Type[BaseModel] = ComparatorInput
    return_direct: bool = True

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
            LLM = get_llm()
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

    def _run(self, cv_directory: str, evaluation_criteria: Dict[str, str]) -> dict:
        """
        Load CVs from directory and apply LLM-based criteria analysis
        """
    
        cv_data = self.read_txt_directory(cv_directory)

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

        for i, cv in enumerate(cv_data.file_contents):
            cv_file_name = cv.file_name
            cv_content = cv.content

            print(f"🔍 Analyzing: {cv_file_name} with LLM...")

            # Use LLM to analyze CV against criteria
            analysis = self._analyze_cv_with_llm(
                cv_content, all_criteria, cv_file_name
            )

            # Calculate scores based on LLM analysis
            raw_score = 0

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
                "filename": cv_file_name,
                "raw_score": raw_score,
                "max_possible_score": max_possible_score,
                "percentage": percentage,
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

        result = json.dumps(
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

        return result

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
        header = "| Rank | Filename | "

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
            row = f"| {rank} | {candidate.get('filename', 'N/A')} | "

            # Add must-have results
            for criterion in (must_have_criteria + nice_to_have_criteria):
                if criterion in candidate.get("llm_analysis", {}).get("criteria_found", []):
                    status = "+"
                else:
                    status = "-"

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

    def read_txt_directory(self, directory: str) -> ReadTxtDirectoryOutput:
        """Read all .txt files from a directory and return their contents."""
        try:
            if not os.path.exists(directory):
                return ReadTxtDirectoryOutput(
                    file_contents=[
                        FileContent(
                            file_name="error",
                            content=f"Error: Directory '{directory}' does not exist.",
                        )
                    ]
                )

            if not os.path.isdir(directory):
                return f"Error: '{directory}' is not a directory."

            file_contents = []
            # Get all .txt files
            for filename in os.listdir(directory):
                if filename.endswith(".txt"):
                    file_path = os.path.join(directory, filename)
                    try:
                        file_contents.append(
                            FileContent(
                                file_name=filename, content=read_txt_file(file_path)
                            )
                        )
                    except Exception as e:
                        file_contents.append(f"Error reading file: {e}")

            return ReadTxtDirectoryOutput(file_contents=file_contents)

        except Exception as e:
            return ReadTxtDirectoryOutput(
                file_contents=[
                    FileContent(
                        file_name="error",
                        content=f"Error accessing directory '{directory}': {str(e)}",
                    )
                ]
            )

