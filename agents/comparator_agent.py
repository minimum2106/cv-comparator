from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Type
import json
import os
from pathlib import Path


# Scoring configuration constants
MAX_OCCURRENCES_FOR_FULL_SCORE = 3
BASE_SCORE_MULTIPLIER = 0.5  # 50% base score for found criteria
FREQUENCY_BONUS_MULTIPLIER = 0.5  # Additional 50% based on frequency
NOT_FOUND_SCORE_MULTIPLIER = 0.1  # 10% base score if criterion not found
PERCENTAGE_MULTIPLIER = 100


class ComparatorInput(BaseModel):
    cv_directory: str = Field(
        description="Directory path containing CV text files to be compared"
    )
    evaluation_criteria: Dict[str, int] = Field(
        description="Evaluation criteria with weights (e.g., {'python': 30, 'experience': 25, 'education': 20, 'projects': 25})"
    )


class ComparatorAgent(BaseTool):
    """A simple agent for comparing CVs against evaluation criteria."""

    name: str = "cv_comparator"
    description: str = (
        "Compare CVs from directory against evaluation criteria and generate scores"
    )
    args_schema: Type[BaseModel] = ComparatorInput


    def _load_cv_files(self, cv_directory: str) -> List[Dict[str, Any]]:
        """Load CV text files from directory and extract content"""
        cv_data = []

        try:
            directory_path = Path(cv_directory)

            if not directory_path.exists():
                raise FileNotFoundError(f"Directory not found: {cv_directory}")

            if not directory_path.is_dir():
                raise ValueError(f"Path is not a directory: {cv_directory}")

            # Find all text files in the directory
            txt_files = list(directory_path.glob("*.txt"))

            if not txt_files:
                raise ValueError(f"No .txt files found in directory: {cv_directory}")

            print(f"📁 Found {len(txt_files)} CV files in directory")

            for txt_file in txt_files:
                try:
                    # Read file content
                    with open(txt_file, "r", encoding="utf-8") as file:
                        content = file.read().strip()

                    if content:
                        cv_info = {
                            "filename": txt_file.name,
                            "name": txt_file.stem,  # Use filename without extension as candidate name
                            "content": content,
                            "file_path": str(txt_file),
                        }
                        cv_data.append(cv_info)
                        print(f"   ✅ Loaded: {txt_file.name}")
                    else:
                        print(f"   ⚠️ Empty file: {txt_file.name}")

                except Exception as e:
                    print(f"   ❌ Error reading {txt_file.name}: {e}")
                    continue

            if not cv_data:
                raise ValueError(
                    "No valid CV content could be loaded from the directory"
                )

            return cv_data

        except Exception as e:
            print(f"❌ Error loading CV files: {e}")
            raise

    def _calculate_criterion_score(
        self, cv_text: str, criterion: str, weight: int
    ) -> float:
        """Calculate score for a single criterion based on occurrences in CV text"""
        criterion_lower = criterion.lower()
        occurrences = cv_text.count(criterion_lower)

        if occurrences > 0:
            # Score based on frequency (max 100% for MAX_OCCURRENCES_FOR_FULL_SCORE+ occurrences)
            frequency_multiplier = min(
                occurrences / MAX_OCCURRENCES_FOR_FULL_SCORE, 1.0
            )
            criterion_score = weight * (
                BASE_SCORE_MULTIPLIER
                + FREQUENCY_BONUS_MULTIPLIER * frequency_multiplier
            )
        else:
            # Base score if criterion not found
            criterion_score = weight * NOT_FOUND_SCORE_MULTIPLIER

        return criterion_score

    def _run(self, cv_directory: str, evaluation_criteria: Dict[str, int]) -> str:
        """
        Load CVs from directory and apply evaluation grid to generate comparison results
        """

        # Load CV files from directory
        cv_data = self._load_cv_files(cv_directory)

        results = []

        for i, cv in enumerate(cv_data):
            candidate_name = cv.get("name", f"Candidate_{i+1}")
            cv_content = cv.get("content", "")

            print(f"🔍 Evaluating: {candidate_name}")

            # Calculate score for each criterion
            scores = {}
            total_score = 0

            cv_text = cv_content.lower()

            for criterion, weight in evaluation_criteria.items():
                criterion_score = self._calculate_criterion_score(
                    cv_text, criterion, weight
                )
                scores[criterion] = round(criterion_score, 2)
                total_score += criterion_score

            # Calculate percentage score
            max_possible_score = sum(evaluation_criteria.values())
            percentage = round(
                (total_score / max_possible_score) * PERCENTAGE_MULTIPLIER, 2
            )

            # Create candidate result
            candidate_result = {
                "name": candidate_name,
                "filename": cv.get("filename"),
                "scores": scores,
                "total_score": round(total_score, 2),
                "percentage": percentage,
            }

            results.append(candidate_result)

        # Sort by total score (highest first)
        results.sort(key=lambda x: x["total_score"], reverse=True)

        # Generate comparison table
        comparison_table = self._generate_comparison_table(results, evaluation_criteria)

        print(f"🏆 Ranking completed for {len(results)} candidates")

        return json.dumps(
            {
                "directory": cv_directory,
                "total_candidates": len(results),
                "ranked_candidates": results,
                "comparison_table": comparison_table,
            },
            indent=2,
        )

    def _generate_comparison_table(
        self, results: List[Dict], criteria: Dict[str, int]
    ) -> str:
        """Generate a markdown comparison table"""

        # Table formatting constants
        DECIMAL_PLACES = 1

        # Table header
        header = "| Rank | Candidate | Filename | "
        header += " | ".join(criteria.keys()) + " | Total Score | Percentage |\n"

        separator = "|------|-----------|----------|"
        separator += (
            "|".join(["----------"] * len(criteria)) + "|------------|------------|\n"
        )

        # Table rows
        rows = []
        for rank, candidate in enumerate(results, 1):
            row = f"| {rank} | {candidate['name']} | {candidate.get('filename', 'N/A')} | "

            # Add scores for each criterion
            criterion_scores = []
            for criterion in criteria.keys():
                score = candidate["scores"].get(criterion, 0)
                criterion_scores.append(f"{score:.{DECIMAL_PLACES}f}")

            row += " | ".join(criterion_scores)
            row += (
                f" | {candidate['total_score']:.{DECIMAL_PLACES}f} | "
                f"{candidate['percentage']:.{DECIMAL_PLACES}f}% |"
            )
            rows.append(row)

        return header + separator + "\n".join(rows)

    def run(self, input_data: Dict[str, Any]) -> str:
        """Run the comparator with input data"""
        return self._run(
            cv_directory=input_data.get("cv_directory", ""),
            evaluation_criteria=input_data.get("evaluation_criteria", {}),
        )
