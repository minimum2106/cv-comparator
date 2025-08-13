from typing import List, Union, Annotated, Dict
import tomllib
import operator
import uuid
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.prompts import GENERATE_PLAN_SYSTEM_PROMPT
from agents.tool_retriever import tool_retriever


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


class TaskList(BaseModel):
    steps: List[TaskStep] = Field(description="List of executable steps")


class OrchestratorState(BaseModel):
    user_query: str
    plan: str = ""
    steps: List[TaskStep] = []
    past_steps: Annotated[List[tuple], operator.add] = []
    response: str = ""
    current_step_index: int = 0  # Remove all_tools


class Orchestrator:
    def __init__(self):
        self.agent_database = {}
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            streaming=False,
        )

    def run(self, user_query: str):
        """Main execution method"""
        print(f"🚀 Starting Orchestrator for query: {user_query}")

        # Initialize state
        initial_state = OrchestratorState(user_query=user_query)

        # Build and execute the orchestrator graph
        graph = self._build_orchestrator_graph()
        final_state = graph.invoke(initial_state)

        # Extract or generate final answer
        if hasattr(final_state, "response") and final_state.response:
            final_answer = final_state.response
        else:
            final_answer = "No response generated."

        return final_answer

    def _build_orchestrator_graph(self):
        """
        Build a workflow graph based on the agents and their execution steps.
        This method would create a StateGraph that represents the workflow.
        """
        graph = StateGraph(OrchestratorState)

        # Simplified flow - remove assign_agent_to_task
        graph.add_node("planning", self._generate_plan)
        graph.add_node("creating_tasks", self._generate_tasks)
        graph.add_node("execute_step", self._execute_current_step)
        graph.add_node("replan", self._replan_execution)

        # Direct flow from task creation to execution
        graph.add_edge(START, "planning")
        graph.add_edge("planning", "creating_tasks")
        graph.add_edge("creating_tasks", "execute_step")  # Direct to execution
        graph.add_edge("execute_step", "replan")
        graph.add_conditional_edges(
            "replan",
            self._should_end_execution,
            {"continue": "execute_step", "end": END},
        )

        return graph.compile()

    def _generate_plan(self, state: OrchestratorState):
        """
        Generate a step-by-step plan based on the user query.
        This method would use an LLM to create a structured plan.
        """

        prompt = f"""
        Analyze the following user query and generate a comprehensive execution plan:

        USER QUERY: {state.user_query}

        Instructions:
        1. Analyze the query complexity and determine the optimal workflow pattern
        2. Break down the objective into executable steps
        3. Ensure each step has clear purpose, inputs, and outputs

        Focus on creating an actionable plan that can be executed by specialized agents and tools to fully accomplish the user's objective.
        """

        messages = [
            SystemMessage(content=GENERATE_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        return {"plan": response.content}

    def _generate_tasks(self, state: OrchestratorState):
        """
        Generate tasks from markdown plan and return list of TaskStep objects.
        """
        plan = state.plan
        tasks = []

        # Extract steps from markdown format
        # Pattern to match: ### Step X: step_name
        step_pattern = r"### Step \d+: ([a-zA-Z_][a-zA-Z0-9_]*)"

        # Split plan into sections by step headers
        step_sections = re.split(step_pattern, plan)[1:]  # Remove first empty element

        # Process pairs of (step_name, content)
        step_pairs = []
        for i in range(0, len(step_sections), 2):
            if i + 1 < len(step_sections):
                step_name = step_sections[i].strip()
                step_content = step_sections[i + 1].strip()
                step_pairs.append((step_name, step_content))

        print(f"📋 Found {len(step_pairs)} steps in plan")

        for step_name, step_content in step_pairs:
            # Extract input from the content
            input_match = re.search(
                r"- \*\*Input\*\*:\s*(.*?)(?=\n-|\n###|$)",
                step_content,
                re.DOTALL,
            )

            # Extract output from the content
            output_match = re.search(
                r"- \*\*Output\*\*:\s*(.*?)(?=\n-|\n###|$)",
                step_content,
                re.DOTALL,
            )

            # Extract description from the content
            description_match = re.search(
                r"- \*\*Description\*\*:\s*(.*?)(?=\n-|\n###|$)",
                step_content,
                re.DOTALL,
            )

            # Extract dependencies from the content
            dependencies_match = re.search(
                r"- \*\*Dependencies\*\*:\s*(.*?)(?=\n-|\n###|$)",
                step_content,
                re.DOTALL,
            )

            # Get input description or fallback
            if input_match:
                step_input = input_match.group(1).strip()
            else:
                step_input = "No input specified"

            # Get output description or fallback
            if output_match:
                step_output = output_match.group(1).strip()
            else:
                step_output = "No output specified"

            # Get description or fallback
            if description_match:
                step_description = description_match.group(1).strip()
            else:
                # Fallback: use first non-empty line as description
                lines = [
                    line.strip() for line in step_content.split("\n") if line.strip()
                ]
                step_description = lines[0] if lines else f"Execute {step_name}"

            # Get dependencies or fallback
            if dependencies_match:
                dependencies_text = dependencies_match.group(1).strip()
                # Parse dependencies - handle "None" or comma-separated values
                if dependencies_text.lower() in ["none", "n/a", ""]:
                    step_dependencies = []
                else:
                    # Split by comma and clean up whitespace
                    step_dependencies = [
                        dep.strip()
                        for dep in dependencies_text.split(",")
                        if dep.strip()
                    ]
            else:
                step_dependencies = []

            # Create TaskStep with all extracted information
            task_step = TaskStep(
                step_id=str(uuid.uuid4()),
                name=step_name,
                description=step_description,
                inputs=step_input,
                outputs=step_output,
                dependencies=step_dependencies,  # Add dependencies
                tools=[],
            )

            tasks.append(task_step)
            print(f"   ✅ {step_name}")

        return {"steps": tasks}

    def _execute_current_step(self, state: OrchestratorState):
        """Execute the current step using React agent with dynamic tool assignment"""

        if not state.steps:
            return {"response": "No steps available for execution"}

        if state.current_step_index >= len(state.steps):
            return {"response": "All steps completed"}

        # Get current step to execute
        current_step = state.steps[state.current_step_index]

        print(f"🚀 Executing Step {state.current_step_index + 1}: {current_step.name}")
        print(f"   Description: {current_step.description}")

        # Dynamic tool assignment for this step
        print(f"🔍 Dynamically retrieving tools for: {current_step.name}")
        tool_docs = tool_retriever.invoke(current_step.description)

        # Extract actual tool objects - only take the first (best) tool
        step_tools = []
        for tool_doc in tool_docs:
            if hasattr(tool_doc, "metadata") and "tool_object" in tool_doc.metadata:
                tool_obj = tool_doc.metadata["tool_object"]
                step_tools.append(tool_obj)

        # Handle case when no tools are available
        if not step_tools:
            print(f"⚠️ No tools available for step: {current_step.name}")
            print(f"🧠 Falling back to LLM reasoning...")
            return self._execute_with_llm_reasoning(current_step, state)

        # Build plan string from all steps for context
        plan_str = "\n".join(
            f"{i + 1}. {step.description}" for i, step in enumerate(state.steps)
        )

        filtered_context = [f"- This is result of {step_name}: {result}" if step_name in current_step.dependencies else "" for step_name, result in state.past_steps]

        task_formatted = f"""For the following plan:
            {plan_str}

            You are tasked with executing step {state.current_step_index + 1}: {current_step.name}
            which you have to execute {current_step.description}
            with the following input's description:
            {current_step.inputs}

            and expected output's description:
            {current_step.outputs}

            Previous completed steps:
            {chr(10).join(filtered_context)}

            Execute this step and provide the result dont modify the result.
            RETURN RAW RESULT !!!
            If you failed to use the tool, dont create any information that is not provided in the plan or previous steps.
        """

        try:
            # Create React agent with dynamically retrieved tools
            agent_executor = create_react_agent(
                self.llm,
                step_tools,
                prompt="""
                    You are a helpful assistant that can use tools to complete tasks.
                """,
            )

            print("-- Executing with tools:")
            print(step_tools)

            agent_response = agent_executor.invoke(
                {"messages": [("user", task_formatted)]}
            )

            result = agent_response["messages"][-1].content
            print(f"✅ Step completed: {result}...")

            return {
                "past_steps": [(current_step.name, result)],
                "current_step_index": state.current_step_index + 1,
            }

        except Exception as e:
            print(f"❌ Step execution failed: {e}")
            print(f"🧠 Falling back to LLM reasoning...")
            return self._execute_with_llm_reasoning(current_step, state)

    def _execute_with_llm_reasoning(
        self, current_step: TaskStep, state: OrchestratorState
    ):
        """Execute step using LLM reasoning when no tools are available"""

        print(f"🧠 No tools available - using LLM reasoning for: {current_step.name}")

        # Build context from previous steps
        context_str = ""
        filtered_context = [f"- {result}" if step_name in current_step.dependencies else "" for step_name, result in state.past_steps]
        if state.past_steps:
            context_str = f"""
                Previous completed steps and their results:
                {chr(10).join(filtered_context)}
            """

        # Create reasoning prompt
        reasoning_prompt = f"""
            You are an expert assistant tasked with completing a workflow step using only your knowledge and reasoning abilities.

            ORIGINAL USER QUERY: {state.user_query}

            CURRENT STEP TO COMPLETE:
            Step Name: {current_step.name}
            Step Description: {current_step.description}

            WORKFLOW CONTEXT:
            {context_str}

            TASK:
            Since no specialized tools are available for this step, please complete it using your knowledge and reasoning.
            Provide a comprehensive response that accomplishes the step objective and moves the workflow forward.

            Be specific, actionable, and ensure your response can be used by subsequent workflow steps.
        """

        try:
            print(f"   🤔 Applying LLM reasoning...")

            response = self.llm.invoke([HumanMessage(content=reasoning_prompt)])
            result = response.content

            print(f"✅ LLM reasoning completed: {result[:100]}...")

            return {
                "past_steps": [(current_step.description, result)],
                "current_step_index": state.current_step_index + 1,
            }

        except Exception as e:
            print(f"❌ LLM reasoning failed: {e}")
            return {
                "past_steps": [
                    (current_step.description, f"LLM reasoning failed: {str(e)}")
                ],
                "current_step_index": state.current_step_index,
            }

    def _replan_execution(self, state: OrchestratorState):
        """Replan the current step if it failed, or continue if successful"""
        print(f"🔄 Replanning based on execution results...")

        # Check if we have completed all steps
        if state.current_step_index >= len(state.steps):
            print("✅ All steps completed - generating final response")
            final_response = self._generate_final_response(state)
            return {"response": final_response}

        # Check if the last step failed
        if state.past_steps:
            last_result = state.past_steps[-1]

            if "Failed:" in last_result or "No tools available" in last_result:
                print(f"❌ Last step failed, replanning...")

                # Get the failed step
                failed_step_index = state.current_step_index
                failed_step = state.steps[failed_step_index]

                # Use LLM to replan this step
                replanned_step = self._replan_failed_step(
                    failed_step, last_result, state.past_steps, state.user_query
                )

                if replanned_step:
                    # Update the step in the steps list
                    new_steps = state.steps.copy()
                    new_steps[failed_step_index] = replanned_step

                    # Reset to retry the failed step
                    print(
                        f"🔄 Updated step {failed_step_index + 1}: {replanned_step.name}"
                    )
                    return {
                        "steps": new_steps,
                        "current_step_index": failed_step_index,  # Reset to retry failed step
                        "past_steps": state.past_steps[
                            :-1
                        ],  # Remove the failed attempt
                    }
                else:
                    print(f"⚠️ Could not replan step, continuing...")
                    return {}
            else:
                print(f"✅ Last step successful, continuing...")
                return {}
        else:
            print(f"📝 No previous steps to evaluate, continuing...")
            return {}

    def _replan_failed_step(
        self,
        failed_step: TaskStep,
        failure_reason: str,
        past_steps: List[tuple],
        user_query: str,
    ) -> TaskStep:
        """Replan a failed step using LLM"""

        replan_prompt = f"""
            A workflow step has failed and needs to be replanned. Analyze the failure and create an improved step.

            ORIGINAL USER QUERY: {user_query}

            FAILED STEP:
            Name: {failed_step.name}
            Description: {failed_step.description}
            Failure Reason: {failure_reason}

            PREVIOUS SUCCESSFUL STEPS:
            {chr(10).join([f"- {step}: {result[:100]}..." for step, result in past_steps[:-1]])}

            Based on the failure reason and context, create an improved version of this step:
            1. Modify the step description to address the failure
            2. Consider alternative approaches
            3. Make the step more specific and actionable
            4. Ensure it builds on previous successful steps

            Respond with:
            STEP_NAME: [improved step name]
            STEP_DESCRIPTION: [improved step description that addresses the failure]
        """

        class AlternateStep(BaseModel):
            step_name: str = Field(description="Improved name for the step")
            step_description: str = Field(
                description="Improved description that addresses the failure"
            )

        try:
            response = self.llm.with_structured_output(AlternateStep).invoke(
                [HumanMessage(content=replan_prompt)]
            )

            # Parse the response

            new_name = response.step_name.stip()
            new_description = response.step_description.strip()

            if new_name and new_description:
                improved_step = TaskStep(
                    step_id=str(uuid.uuid4()),
                    name=new_name,
                    description=new_description,
                    inputs=failed_step.inputs,
                    outputs=failed_step.outputs,
                    tools=[],  # Tools will be assigned dynamically
                )

                return improved_step
            else:
                print(f"   ❌ Could not parse replanned step")
                return None

        except Exception as e:
            print(f"   ❌ Replanning failed: {e}")
            return None

    def _should_end_execution(self, state: OrchestratorState):
        """Determine if execution should end"""
        # End if we have a final response
        if hasattr(state, "response") and state.response:
            return "end"

        # End if we've completed all steps
        if state.current_step_index >= len(state.steps):
            return "end"

        # Continue if there are more steps
        return "continue"

    def _generate_final_response(self, state: OrchestratorState):
        """Generate final response when all steps are completed"""

        if not state.past_steps:
            return "No steps were executed successfully."

        final_prompt = f"""
        Based on the execution of all workflow steps, provide a comprehensive final answer:

        Original Query: {state.user_query}

        Completed Steps:
        {chr(10).join([f"{i+1}. {step}: {result}" for i, (step, result) in enumerate(state.past_steps)])}

        Provide a complete answer to the original user query, incorporating insights from all completed steps.
        """

        response = self.llm.invoke([HumanMessage(content=final_prompt)])
        return response.content
