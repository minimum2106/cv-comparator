import os
import sys
from typing import List


class UIManager:
    """Handles all terminal UI and printing operations for the orchestrator"""

    def __init__(self):
        self.width = 70

    def clear_screen(self):
        """Clear terminal screen for better visual experience"""
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self, title: str, emoji: str = "🤖"):
        """Print a styled header with orchestrator summary"""
        print("\n" + "═" * self.width)
        print(f"{emoji} {title:^{self.width-4}} {emoji}")
        print("═" * self.width)

        # Add orchestrator summary
        if title == "CV Comparator Assistant":
            summary_lines = [
                "🎯 AI-Powered Workflow Orchestrator",
                "",
                "✨ What I can do:",
                "  • Compare CVs against job requirements",
                "  • Analyze documents and extract insights",
                "  • Process data and generate reports",
                "  • Execute multi-step workflows automatically",
                "",
                "🔧 How it works:",
                "  1. Validate your request for completeness",
                "  2. Generate an execution plan",
                "  3. Execute each step with specialized tools",
                "  4. Provide comprehensive results",
                "",
                "💡 Tips:",
                "  • Be specific about file paths and requirements",
                "  • Type 'quit' anytime to exit the program",
                "  • I'll ask clarifying questions if needed",
            ]

            for line in summary_lines:
                if line:
                    print(f"│ {line:<{self.width-4}} │")
                else:
                    print(f"│{' ' * (self.width-2)}│")

        print("═" * self.width)

    def print_assistant_message(self, message: str):
        """Print assistant message with styling"""
        print("\n💬 Assistant:")
        print("┌" + "─" * 68 + "┐")

        # Split message into lines and wrap long lines
        lines = message.split("\n")
        for line in lines:
            if len(line) <= 66:
                print(f"│ {line:<66} │")
            else:
                # Simple word wrap
                words = line.split(" ")
                current_line = ""
                for word in words:
                    if len(current_line + word) <= 64:
                        current_line += word + " "
                    else:
                        print(f"│ {current_line.strip():<66} │")
                        current_line = word + " "
                if current_line.strip():
                    print(f"│ {current_line.strip():<66} │")

        print("└" + "─" * 68 + "┘")

    def print_user_prompt(self):
        """Print user input prompt with styling"""
        print("\n👤 You:")
        print("┌" + "─" * 68 + "┐")
        print("│ Type your response below (or 'quit' to exit):                 │")
        print("└" + "─" * 68 + "┘")
        print("💭 ", end="", flush=True)

    def format_issues_and_questions(
        self, issues: List[str], questions: List[str]
    ) -> str:
        """Format issues and questions in a conversation-friendly way"""
        message = "I need more information to help you properly.\n\n"

        if issues:
            message += "🔍 Issues I found with your request:\n"
            for i, issue in enumerate(issues, 1):
                message += f"   {i}. {issue}\n"
            message += "\n"

        if questions:
            message += "❓ Please help me understand:\n"
            for i, question in enumerate(questions, 1):
                message += f"   {i}. {question}\n"

        return message

    def print_goodbye_message(self):
        """Print styled goodbye message and exit"""
        self.clear_screen()
        self.print_header("Goodbye!", "👋")

        exit_message = """Thank you for using the CV Comparator Assistant!

🎯 Session Summary:
  • Workflow execution stopped by user request
  • No analysis was completed
  • All data and progress have been cleared

💡 Next time:
  • Provide specific file paths for better results
  • Include clear objectives in your requests
  • I'm here whenever you need workflow automation!

Have a great day! 🌟"""

        self.print_assistant_message(exit_message)
        print("\n" + "═" * self.width)
        print("🚪 Exiting CV Comparator Assistant...")
        print("═" * self.width)

    def print_analysis_start(self, query: str):
        """Print analysis start message"""
        print(f"\n🚀 Starting analysis for your request...")
        print(f"📝 Your query: {query}")

    def print_processing_message(self):
        """Print processing message"""
        print("\n⏳ Processing your response...")

    def print_completion_message(self):
        """Print completion message"""
        print("\n" + "═" * self.width)
        print("🎉 Analysis completed successfully!")
        print("💡 Run the program again for a new analysis")
        print("═" * self.width)

    def print_step_execution(self, step_index: int, step_name: str, description: str):
        """Print step execution information"""
        print("┌" + "─" * 68 + "┐")
        print(f"│ Executing Step {step_index + 1}: {step_name} │")
        print("└" + "─" * 68 + "┘")
        print(f"   Description: {description}")

    def print_tool_retrieval(self, step_name: str):
        """Print tool retrieval message"""
        print(f"🔍 Dynamically retrieving tools for: {step_name}")

    def print_tools_found(self, tool_count: int, tool_names: List[str]):
        """Print tools found message"""
        if tool_names:
            print(f"   🛠️ Found {tool_count} tool(s): {tool_names}")
        else:
            print(f"   🛠️ Found {tool_count} tool(s): None - will use LLM reasoning")

    def print_tool_execution(self, tool_count: int):
        """Print tool execution message"""
        print(f"   🔧 Executing with {tool_count} tool(s)")

    def print_step_completed(self, result: str):
        """Print step completion message"""
        print(f"✅ Step completed")
        print(f"📄 Result: {result}...")

    def print_step_failed(self):
        """Print step failure message"""
        print("❌ Step execution failed")

    def print_step_exception(self, error: str):
        """Print step exception message"""
        print(f"❌ Step execution failed with exception: {error}")

    def print_replanning(self):
        """Print replanning message"""
        print(f"🔄 Replanning based on execution results...")

    def print_all_steps_completed(self):
        """Print all steps completed message"""
        print("✅ All steps completed - generating final response")

    def print_failed_step_replanning(self):
        """Print failed step replanning message"""
        print(f"❌ Last step failed, replanning...")

    def print_step_updated(self, step_index: int, step_name: str):
        """Print step updated message"""
        print(f"🔄 Updated step {step_index + 1}: {step_name}")

    def print_replan_failed(self):
        """Print replan failed message"""
        print(f"⚠️ Could not replan step, continuing...")

    def print_last_step_successful(self):
        """Print last step successful message"""
        print(f"✅ Last step successful, continuing...")

    def print_no_previous_steps(self):
        """Print no previous steps message"""
        print(f"📝 No previous steps to evaluate, continuing...")

    def print_replan_parse_error(self):
        """Print replan parse error message"""
        print(f"   ❌ Could not parse replanned step")

    def print_replan_exception(self, error: str):
        """Print replan exception message"""
        print(f"   ❌ Replanning failed: {error}")

    def print_plan_found(self, step_count: int):
        """Print plan found message"""
        print(f"📋 Found {step_count} steps in plan")

    def print_step_created(self, step_name: str):
        """Print step created message"""
        print(f"   ✅ {step_name}")

    def check_quit_command(self, user_input: str) -> bool:
        """Check if user input is a quit command"""
        return user_input.lower() in ["quit", "exit", "q"]
