from typing import Dict, List
from dotenv import load_dotenv
import uuid
import json

from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from agents.writer_agent import WriterAgent
from agents.comparator_agent import ComparatorAgent
from agents.scorecard_agent import ScorecardAgent


load_dotenv()

class QueryForTools(BaseModel):
    """Generate a query for additional tools."""

    query: str = Field(..., description="Query for additional tools.")


class ToolRag:
    def __init__(self):
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.0,
            streaming=False,
        )
        self.vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings())
        self.tool_registry = {}

    def add_tool(self, tool: StructuredTool):
        """Add a tool to the RAG system."""
        # Implementation to add tool metadata to RA
        tool_id = str(uuid.uuid4())
        self.tool_registry[tool_id] = tool
        tool_document = Document(
            page_content=tool.description,
            id=tool_id,
            metadata={"tool_name": tool.name},
        )
        self.vector_store.add_documents([tool_document])

    def get_tool(self, tool_name: str) -> StructuredTool:
        """Retrieve a tool by name."""
        # Implementation to fetch tool metadata from RA
        for tool in self.tool_registry.values():
            if tool.name == tool_name:
                return tool

        raise ValueError(f"Tool '{tool_name}' not found in RAG system.")

    def get_tools_by_query(self, query: str) -> list[StructuredTool]:
        """Search for tools based on a query."""
        # Implementation to search tools in RAG

        # generate multi-query from orginal query for better results

        # Example: if query is "file reader", generate queries like "file reader tool", "file reading agent", etc.

        # using reranking to get the best results

        system = SystemMessage(
            "Given this conversation, generate a query for additional tools. "
            "The query should be a short string containing what type of information "
            "is needed. If no further information is needed, "
            "set more_information_needed False and populate a blank string for the query."
        )
        input_messages = [system] + [HumanMessage(
            f"""
            From this query, could you give me a list of tools that can help with this query?
            Query: {query}
            """
        )]
        response = self.llm.bind_tools([QueryForTools], tool_choice=True).invoke(
            input_messages
        )
        query = response.tool_calls[0]["args"]["query"]

        tool_documents = self.vector_store.similarity_search(query, k=1 )
        selected_tools = [
            self.get_tool(document.metadata["tool_name"])
            for document in tool_documents
        ]

        # rerank the tools based on their relevance to the query using reranking model
        # This is a placeholder for the reranking logic
        # selected_tools = rerank_tools(selected_tools, query)

        
        return selected_tools


# 1. CV File Reader Tool
def cv_file_reader(file_path: str) -> str:
    """Read CV files in various formats (PDF, DOCX, TXT)"""
    try:
        # Simplified implementation (would use real parsers in production)
        return f"Successfully read CV from {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


# 2. Skill Extractor Tool
def skill_extractor(
    cv_text: str, skill_categories: str = "technical,soft,languages"
) -> str:
    """Extract skills from CV text and categorize them"""
    # Simplified implementation
    skills = {
        "technical": ["Python", "SQL", "AWS"],
        "soft": ["Communication", "Leadership"],
        "languages": ["English", "French"],
    }
    return json.dumps(skills)


# 3. CV Comparison Tool
def cv_comparison(cv_list: list, job_requirements: str) -> str:
    """Compare multiple CVs against job requirements and rank candidates"""
    # Simplified implementation
    rankings = [
        {
            "name": "Candidate 1",
            "match_score": 0.85,
            "strengths": ["Strong technical skills"],
        },
        {"name": "Candidate 2", "match_score": 0.72, "strengths": ["Good culture fit"]},
    ]
    return json.dumps({"rankings": rankings, "total_compared": len(cv_list)})


# 4. Report Generator Tool
def report_generator(comparison_data: dict, format_type: str = "markdown") -> str:
    """Generate professional reports from CV comparison data"""
    # Simplified implementation
    return """
    # CV Comparison Report
    
    ## Top Candidates
    1. Candidate 1 (85% match)
    2. Candidate 2 (72% match)
    
    ## Analysis
    The candidates have been ranked based on their match to the job requirements.
    """


# 5. Email Notification Tool
def email_sender(recipient: str, subject: str, body: str) -> str:
    """Send email with CV analysis results"""
    # Simplified implementation
    return f"Email sent to {recipient} with subject '{subject}'"


# Create StructuredTools
cv_reader_tool = StructuredTool.from_function(
    func=cv_file_reader,
    name="cv_file_reader",
    description="Reads CV files in various formats (PDF, DOCX, TXT) and extracts their text content for analysis",
)

skill_extractor_tool = StructuredTool.from_function(
    func=skill_extractor,
    name="skill_extractor",
    description="Analyzes CV text to extract and categorize skills like programming languages, soft skills, and languages",
)

cv_comparison_tool = StructuredTool.from_function(
    func=cv_comparison,
    name="cv_comparison",
    description="Compares multiple candidate CVs against job requirements and produces rankings with match scores",
)

report_tool = StructuredTool.from_function(
    func=report_generator,
    name="report_generator",
    description="Creates professional client-ready reports from CV comparison data in various formats",
)

email_tool = StructuredTool.from_function(
    func=email_sender,
    name="email_sender",
    description="Sends email notifications with CV analysis results and reports to clients or hiring managers",
)


# Initialize and populate ToolRag
def setup_tool_rag():
    tool_rag = ToolRag()

    # Add tools
    for tool in [
        cv_reader_tool,
        skill_extractor_tool,
        cv_comparison_tool,
        report_tool,
        email_tool,
        WriterAgent(),
        ComparatorAgent(),
        ScorecardAgent(),
    ]:
        tool_rag.add_tool(tool)
        print(f"Added tool: {tool.name}")

    return tool_rag

tool_rag = setup_tool_rag()