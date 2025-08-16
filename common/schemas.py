from typing import List, Union

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class TaskStep(BaseModel):
    step_id: str = Field(description="Unique identifier for the step")
    name: str = Field(description="Name of the step")
    description: str = Field(
        description="Detailed description of what this step accomplishes"
    )
    inputs: Union[str, None] = Field(
        default="", description="Descriptions of the inputs for this step"
    )
    outputs: Union[str, None] = Field(
        default="", description="Descriptions of the expected outputs for this step"
    )
    tools: List[BaseTool] = Field(
        default_factory=list, description="List of tools required for this step"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="List of step names that this step depends on"
    )

