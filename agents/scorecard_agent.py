from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Type
import json


class ScorecardInput(BaseModel):
    job_brief: str = Field(
        description="HR job brief describing the position requirements"
    )
    position_title: str = Field(description="Job title/position name")
    custom_criteria: List[str] = Field(
        default=[], description="Optional custom criteria to include in the scorecard"
    )


class ScorecardAgent(BaseTool):
    """A simple agent for generating evaluation scorecards from job briefs."""

    name: str = "scorecard_generator"
    description: str = "Generate evaluation scorecard from HR job brief"
    args_schema: Type[BaseModel] = ScorecardInput

    def _run(
        self, job_brief: str, position_title: str, custom_criteria: List[str] = None
    ) -> str:
        """
        Create evaluation scorecard from job brief
        """
        if custom_criteria is None:
            custom_criteria = []

        # Extract criteria from job brief
        extracted_criteria = self._extract_criteria_from_brief(
            job_brief, position_title
        )

        # Add custom criteria if provided
        if custom_criteria:
            for criterion in custom_criteria:
                if criterion not in extracted_criteria:
                    extracted_criteria[criterion] = 15  # Default weight

        # Normalize weights to total 100
        criteria_weights = self._normalize_weights(extracted_criteria)

        # Generate scorecard format
        scorecard = {
            "position": position_title,
            "criteria": criteria_weights,
            "total_weight": sum(criteria_weights.values()),
            "scorecard_format": self._generate_scorecard_format(criteria_weights),
            "evaluation_guide": self._generate_evaluation_guide(criteria_weights),
        }

        return json.dumps(scorecard, indent=2)

    def _extract_criteria_from_brief(
        self, job_brief: str, position_title: str
    ) -> Dict[str, int]:
        """Extract evaluation criteria from job brief"""
        brief_lower = job_brief.lower()
        position_lower = position_title.lower()

        # Base criteria with default weights
        base_criteria = {}

        # Technical skills based on job brief content
        if "python" in brief_lower or "python" in position_lower:
            base_criteria["python_skills"] = 25
        if "java" in brief_lower:
            base_criteria["java_skills"] = 25
        if "javascript" in brief_lower or "js" in brief_lower:
            base_criteria["javascript_skills"] = 25
        if "react" in brief_lower or "vue" in brief_lower or "angular" in brief_lower:
            base_criteria["frontend_frameworks"] = 20
        if "django" in brief_lower or "flask" in brief_lower or "spring" in brief_lower:
            base_criteria["backend_frameworks"] = 20
        if (
            "sql" in brief_lower
            or "database" in brief_lower
            or "mysql" in brief_lower
            or "postgresql" in brief_lower
        ):
            base_criteria["database_skills"] = 15
        if (
            "docker" in brief_lower
            or "kubernetes" in brief_lower
            or "aws" in brief_lower
            or "cloud" in brief_lower
        ):
            base_criteria["devops_cloud"] = 15

        # Experience level
        if "senior" in brief_lower or "lead" in brief_lower:
            base_criteria["experience_level"] = 20
        elif "junior" in brief_lower or "entry" in brief_lower:
            base_criteria["experience_level"] = 15
        else:
            base_criteria["experience_level"] = 18

        # Soft skills
        if "team" in brief_lower or "collaboration" in brief_lower:
            base_criteria["teamwork"] = 10
        if "communication" in brief_lower:
            base_criteria["communication"] = 10
        if "leadership" in brief_lower or "management" in brief_lower:
            base_criteria["leadership"] = 15

        # Education
        if (
            "degree" in brief_lower
            or "university" in brief_lower
            or "education" in brief_lower
        ):
            base_criteria["education"] = 10

        # If no specific criteria found, use defaults
        if not base_criteria:
            base_criteria = {
                "technical_skills": 30,
                "experience_level": 25,
                "education": 15,
                "communication": 10,
                "problem_solving": 10,
                "cultural_fit": 10,
            }

        return base_criteria

    def _normalize_weights(self, criteria: Dict[str, int]) -> Dict[str, int]:
        """Normalize weights to total 100"""
        current_total = sum(criteria.values())

        if current_total == 0:
            return criteria

        # Scale to 100
        normalized = {}
        for criterion, weight in criteria.items():
            normalized[criterion] = round((weight / current_total) * 100)

        # Adjust for rounding errors
        total_normalized = sum(normalized.values())
        if total_normalized != 100:
            # Add/subtract difference to the largest weight
            largest_key = max(normalized.keys(), key=lambda k: normalized[k])
            normalized[largest_key] += 100 - total_normalized

        return normalized

    def _generate_scorecard_format(self, criteria: Dict[str, int]) -> str:
        """Generate scorecard format as markdown table"""

        header = "| Criterion | Weight | Score (1-10) | Weighted Score | Notes |\n"
        separator = "|-----------|--------|--------------|----------------|-------|\n"

        rows = []
        for criterion, weight in criteria.items():
            criterion_display = criterion.replace("_", " ").title()
            row = f"| {criterion_display} | {weight}% | ___ | ___ | ___ |"
            rows.append(row)

        total_row = "| **TOTAL** | **100%** | **___** | **___** | **___** |"

        return header + separator + "\n".join(rows) + "\n" + total_row

    def _generate_evaluation_guide(self, criteria: Dict[str, int]) -> Dict[str, str]:
        """Generate evaluation guide for each criterion"""

        guides = {}

        evaluation_guides = {
            "python_skills": "Evaluate Python proficiency, frameworks knowledge, and coding best practices",
            "java_skills": "Assess Java programming skills, OOP concepts, and enterprise frameworks",
            "javascript_skills": "Review JS/TS skills, ES6+ features, and modern development practices",
            "frontend_frameworks": "Evaluate React/Vue/Angular experience and component-based development",
            "backend_frameworks": "Assess Django/Flask/Spring experience and API development",
            "database_skills": "Review SQL knowledge, database design, and data management skills",
            "devops_cloud": "Evaluate Docker, Kubernetes, CI/CD, and cloud platform experience",
            "experience_level": "Assess years of experience, project complexity, and technical depth",
            "teamwork": "Evaluate collaboration skills, team projects, and interpersonal abilities",
            "communication": "Assess written/verbal communication and documentation skills",
            "leadership": "Review mentoring, project leadership, and decision-making experience",
            "education": "Evaluate formal education, certifications, and continuous learning",
            "technical_skills": "Overall technical competency and problem-solving ability",
            "problem_solving": "Assess analytical thinking and troubleshooting capabilities",
            "cultural_fit": "Evaluate alignment with company values and work style",
        }

        for criterion in criteria.keys():
            guides[criterion] = evaluation_guides.get(
                criterion,
                f"Evaluate candidate's {criterion.replace('_', ' ')} based on job requirements",
            )

        return guides

    def run(self, input_data: Dict[str, Any]) -> str:
        """Run the scorecard generator with input data"""
        return self._run(
            job_brief=input_data.get("job_brief", ""),
            position_title=input_data.get("position_title", ""),
            custom_criteria=input_data.get("custom_criteria", []),
        )
