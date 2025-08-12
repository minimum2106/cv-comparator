from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Type
import json


class ComparatorInput(BaseModel):
    cv_data: List[Dict[str, Any]] = Field(
        description="List of CV data with extracted information"
    )
    evaluation_criteria: Dict[str, int] = Field(
        description="Evaluation criteria with weights (e.g., {'python': 30, 'experience': 25, 'education': 20, 'projects': 25})"
    )


class ComparatorAgent(BaseTool):
    """A simple agent for comparing CVs against evaluation criteria."""

    name: str = "cv_comparator"
    description: str = "Compare CVs against evaluation criteria and generate scores"
    args_schema: Type[BaseModel] = ComparatorInput

    def _run(
        self, cv_data: List[Dict[str, Any]], evaluation_criteria: Dict[str, int]
    ) -> str:
        """
        Apply evaluation grid to CVs and generate comparison results
        """
        results = []

        for i, cv in enumerate(cv_data):
            candidate_name = cv.get("name", f"Candidate_{i+1}")

            # Calculate score for each criterion
            scores = {}
            total_score = 0

            for criterion, weight in evaluation_criteria.items():
                # Simple scoring logic based on keyword presence
                cv_text = str(cv).lower()

                if criterion.lower() in cv_text:
                    # Basic scoring: if criterion found, give proportional score
                    criterion_score = weight * 0.8  # 80% if found
                else:
                    criterion_score = weight * 0.2  # 20% if not found

                scores[criterion] = criterion_score
                total_score += criterion_score

            # Create candidate result
            candidate_result = {
                "name": candidate_name,
                "scores": scores,
                "total_score": round(total_score, 2),
                "percentage": round(
                    (total_score / sum(evaluation_criteria.values())) * 100, 2
                ),
            }

            results.append(candidate_result)

        # Sort by total score (highest first)
        results.sort(key=lambda x: x["total_score"], reverse=True)

        # Generate comparison table
        comparison_table = self._generate_comparison_table(results, evaluation_criteria)

        return json.dumps(
            {"ranked_candidates": results, "comparison_table": comparison_table},
            indent=2,
        )

    def _generate_comparison_table(
        self, results: List[Dict], criteria: Dict[str, int]
    ) -> str:
        """Generate a markdown comparison table"""

        # Table header
        header = "| Rank | Candidate | "
        header += " | ".join(criteria.keys()) + " | Total Score | Percentage |\n"

        separator = "|------|-----------|"
        separator += (
            "|".join(["----------"] * len(criteria)) + "|------------|------------|\n"
        )

        # Table rows
        rows = []
        for rank, candidate in enumerate(results, 1):
            row = f"| {rank} | {candidate['name']} | "

            # Add scores for each criterion
            criterion_scores = []
            for criterion in criteria.keys():
                score = candidate["scores"].get(criterion, 0)
                criterion_scores.append(f"{score:.1f}")

            row += " | ".join(criterion_scores)
            row += (
                f" | {candidate['total_score']:.1f} | {candidate['percentage']:.1f}% |"
            )
            rows.append(row)

        return header + separator + "\n".join(rows)

    def run(self, input_data: Dict[str, Any]) -> str:
        """Run the comparator with input data"""
        return self._run(
            cv_data=input_data.get("cv_data", []),
            evaluation_criteria=input_data.get("evaluation_criteria", {}),
        )
