import os
import sys
from typing import List, Union, Annotated, Dict
from dotenv import load_dotenv
import tomllib
import operator
import uuid
import re

from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.prompts import GENERATE_PLAN_SYSTEM_PROMPT, VALIDATE_USER_QUERY
from retrievers.tool_retriever import tool_retriever
from common.ui_manager import UIManager
from common.schemas import TaskStep


load_dotenv()


class OrchestratorState(BaseModel):
    user_query: str
    plan: str = ""
    steps: List[TaskStep] = []
    past_steps: Annotated[List[tuple], operator.add] = []
    response: str = ""
    current_step_index: int = 0


class Orchestrator:
    def __init__(self):
        self.agent_database = {}
        self.ui = UIManager()

        with open("project.toml", "rb") as f:
            config = tomllib.load(f)
            provider = config.get("project", {}).get("models").get("provider")

            if provider == "openai":
                model = config.get("project", {}).get("models").get("openai_default")
                self.llm = ChatOpenAI(model=model, temperature=0.0, streaming=False)
            elif provider == "groq":
                model = config.get("project", {}).get("models").get("groq_default")
                self.llm = ChatGroq(model=model, temperature=0.0, streaming=False)
            else:
                raise ValueError("Unsupported model provider")

    def _handle_quit(self):
        """Handle user quit command with styled exit message"""
        self.ui.print_goodbye_message()
        sys.exit(0)

    def run(self):
        """Main execution method with enhanced UI"""
        self.ui.clear_screen()
        self.ui.print_header("CV Comparator Assistant", "🎯")

        self.ui.print_assistant_message(
            "Welcome! Please describe what you'd like me to help you with today."
        )

        self.ui.print_user_prompt()
        user_input = self.get_multiline_input()

        # Check for quit command
        if self.ui.check_quit_command(user_input):
            self._handle_quit()

        self.ui.print_analysis_start(user_input)

        # Initialize state
        initial_state = OrchestratorState(user_query=user_input)

        # Build and execute the orchestrator graph
        graph = self._build_orchestrator_graph()

        config = {"configurable": {"thread_id": uuid.uuid4()}}
        final_state = graph.invoke(initial_state, config=config)

        # Enhanced interrupt handling with conversation UI
        while final_state.get("__interrupt__"):
            self.ui.clear_screen()
            self.ui.print_header("Need More Information", "💬")

            # Extract the question from interrupt
            interrupt_data = final_state["__interrupt__"][-1]
            question_text = (
                interrupt_data.value
                if hasattr(interrupt_data, "value")
                else str(interrupt_data)
            )

            # Format and display the assistant's message
            self.ui.print_assistant_message(question_text)

            # Get user input with styled prompt
            self.ui.print_user_prompt()
            current_user_input = self.get_multiline_input()

            # Check for quit command during interaction
            if self.ui.check_quit_command(current_user_input):
                self._handle_quit()

            user_input += "\n\nAdditional information: " + current_user_input

            # Show loading message
            self.ui.print_processing_message()

            # Clear interrupt and resume
            final_state["__interrupt__"] = []
            final_state = graph.invoke(Command(resume=user_input), config=config)

        # Clear screen for final result
        self.ui.clear_screen()
        self.ui.print_header("Analysis Complete", "✅")

        # Extract or generate final answer
        if hasattr(final_state, "response") and final_state.response:
            final_answer = final_state.response
        else:
            final_answer = "No response generated."

        # Display final result with styling
        self.ui.print_assistant_message(final_answer)
        self.ui.print_completion_message()

        return final_answer

    def _build_orchestrator_graph(self):
        """
        Build a workflow graph based on the agents and their execution steps.
        This method would create a StateGraph that represents the workflow.
        """
        checkpointer = InMemorySaver()
        graph = StateGraph(OrchestratorState)

        # Simplified flow - remove assign_agent_to_task
        graph.add_node("validate_query", self._validate_user_query)
        graph.add_node("planning", self._generate_plan)
        graph.add_node("creating_tasks", self._generate_tasks)
        graph.add_node("execute_step", self._execute_current_step)
        graph.add_node("replan", self._replan_execution)

        graph.add_edge(START, "validate_query")
        graph.add_edge("validate_query", "planning")
        graph.add_edge("planning", "creating_tasks")
        graph.add_edge("creating_tasks", "execute_step")  # Direct to execution
        graph.add_edge("execute_step", "replan")
        graph.add_conditional_edges(
            "replan",
            self._should_end_execution,
            {"continue": "execute_step", "end": END},
        )

        return graph.compile(checkpointer=checkpointer)

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

        self.ui.print_plan_found(len(step_pairs))

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
            self.ui.print_step_created(step_name)

        return {"steps": tasks}

    def _execute_current_step(self, state: OrchestratorState):
        """Execute the current step using React agent with dynamic tool assignment"""

        if not state.steps:
            return {"response": "No steps available for execution"}

        if state.current_step_index >= len(state.steps):
            return {"response": "All steps completed"}

        # Get current step to execute
        current_step = state.steps[state.current_step_index]

        self.ui.print_step_execution(
            state.current_step_index, current_step.name, current_step.description
        )

        # Dynamic tool assignment for this step
        self.ui.print_tool_retrieval(current_step.name)
        tool_docs = tool_retriever.invoke(current_step.description)

        # Extract actual tool objects
        step_tools = []
        for tool_doc in tool_docs:
            if hasattr(tool_doc, "metadata") and "tool_object" in tool_doc.metadata:
                tool_obj = tool_doc.metadata["tool_object"]
                step_tools.append(tool_obj)

        tool_names = [tool.name for tool in step_tools] if step_tools else []
        self.ui.print_tools_found(len(step_tools), tool_names)

        # Build plan string from all steps for context
        plan_str = "\n".join(
            f"{i + 1}. {step.description}" for i, step in enumerate(state.steps)
        )

        # Filter context based on dependencies
        filtered_context = []
        for step_name, result in state.past_steps:
            if step_name in current_step.dependencies:
                filtered_context.append(f"- {step_name}: {result}")

        task_formatted = f"""
            You are tasked with executing step: {current_step.name}
            Description: {current_step.description}

            Input requirements: {current_step.inputs}
            Expected outputs: {current_step.outputs}

            Previous completed steps (dependencies) this is where you could get the context you need to call tool if you have any:
            {chr(10).join(filtered_context) if filtered_context else "No dependencies"}

            Execute this step and return the result
        """

        try:

            agent_executor = create_react_agent(
                self.llm,
                step_tools,  # Empty list is fine - agent will use LLM reasoning
                prompt="""
                    You are a helpful assistant that executes a workflow step.
                    Please focus on the step provided and its requirements.

                    ONLY FOCUS ON SOLVING THAT STEP, NOTHING ELSE.
                    WHEN YOU FINISH THE INDICATED STEP, RETURN ITS RESULT AND STOP WORKING

                    RULES:
                    1. If tools are available, use them to complete the task
                    2. If no tools are available, use your knowledge and reasoning
                    3. Always try to complete the step as requested
                    4. Return actual results, not explanations of what you would do
                    5. If you cannot complete the step, return <FAILED_STEP> with detailed explanation about the error

                    Be direct and provide actionable results.
                    IMPORTANT: After you get a tool result, DO NOT call any tool again. Just return the result and finish the step.
                """,
            )

            self.ui.print_tool_execution(len(step_tools))

            agent_response = agent_executor.invoke(
                {"messages": [("user", task_formatted)]}
            )

            result = agent_response["messages"][-1].content
            self.ui.print_step_completed(result)

            if "<FAILED_STEP>" in result:
                self.ui.print_step_failed()
                return {
                    "past_steps": [(current_step.name, result)],
                    "current_step_index": state.current_step_index,  # Don't advance
                }

            return {
                "past_steps": [(current_step.name, result)],
                "current_step_index": state.current_step_index + 1,
            }

        except Exception as e:
            self.ui.print_step_exception(str(e))
            error_result = f"<FAILED_STEP>: Exception occurred - {str(e)}"
            return {
                "past_steps": [(current_step.name, error_result)],
                "current_step_index": state.current_step_index,  # Don't advance
            }

    def _replan_execution(self, state: OrchestratorState):
        """Replan the current step if it failed, or continue if successful"""
        self.ui.print_replanning()

        # Check if we have completed all steps
        if state.current_step_index >= len(state.steps):
            self.ui.print_all_steps_completed()
            final_response = self._generate_final_response(state)
            return {"response": final_response}

        # Check if the last step failed
        if state.past_steps:
            last_result = state.past_steps[-1]

            if "<FAILED_STEP>" in last_result[1]:
                self.ui.print_failed_step_replanning()

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
                    self.ui.print_step_updated(failed_step_index, replanned_step.name)
                    return {
                        "steps": new_steps,
                        "current_step_index": failed_step_index,  # Reset to retry failed step
                        "past_steps": state.past_steps[
                            :-1
                        ],  # Remove the failed attempt
                    }
                else:
                    self.ui.print_replan_failed()
                    return {}
            else:
                self.ui.print_last_step_successful()
                return {}
        else:
            self.ui.print_no_previous_steps()
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

            new_name = response.step_name.strip()
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
                self.ui.print_replan_parse_error()
                return None

        except Exception as e:
            self.ui.print_replan_exception(str(e))
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

    def _validate_user_query(self, state: OrchestratorState):
        """Validate the user's query before processing"""
        user_query = state.user_query


        class QueryValidation(BaseModel):
            validation_result: str = Field(
                description="Either 'SUFFICIENT' or 'INSUFFICIENT'"
            )
            issues: List[str] = Field(description="List of problems with the query")
            clarifying_questions: List[str] = Field(
                description="Questions to ask if insufficient", default=[]
            )
            plan_preview: List[str] = Field(
                description="Brief workflow steps if sufficient", default=[]
            )

        structured_llm = self.llm.with_structured_output(QueryValidation)
        system_prompt = VALIDATE_USER_QUERY.format(user_query=user_query)
        response = structured_llm.invoke([HumanMessage(content=system_prompt)])

        validation_result = response.validation_result

        while validation_result == "INSUFFICIENT":
            # Format the conversation message
            conversation_message = self.ui.format_issues_and_questions(
                response.issues, response.clarifying_questions
            )
            conversation_message += (
                "\n💡 Please provide additional details to help me assist you better."
            )

            # Use interrupt to ask for more information
            user_query = interrupt(conversation_message)

            # Re-validate with the additional information
            updated_prompt = VALIDATE_USER_QUERY.format(user_query=user_query)
            response = structured_llm.invoke([HumanMessage(content=updated_prompt)])
            validation_result = response.validation_result

        # Clean up the final query
        cleaned_user_query = self.llm.invoke(
            [
                SystemMessage(
                    content="""
                You are a query reorganization expert. 
                Your task is to create a cleaned version of the following user input.
                REMEMBER TO NOT REMOVE OR ADD ANY INFORMATION FROM THE USER INPUT, JUST REWRITE TO BE EASIER TO UNDERSTAND.
                Make it concise and clear while preserving all the important details.
            """
                ),
                HumanMessage(content=user_query),
            ]
        )

        return {"user_query": cleaned_user_query.content}

    def get_multiline_input(self) -> str:
        """Get multiline input from user, ending when they press Enter twice"""
        lines = []
        empty_line_count = 0

        while True:
            try:
                line = input()

                # Check if it's an empty line
                if line.strip() == "":
                    empty_line_count += 1
                    # If this is the second consecutive empty line, break
                    if empty_line_count >= 2:
                        break
                    lines.append(line)
                else:
                    empty_line_count = 0
                    lines.append(line)

            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n🚫 Input cancelled. Type 'quit' to exit.")
                return ""

        # Remove trailing empty lines
        while lines and lines[-1].strip() == "":
            lines.pop()

        return "\n".join(lines)
