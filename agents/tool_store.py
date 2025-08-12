from typing import Dict, List
from dotenv import load_dotenv
import uuid
import json

from langchain.tools import StructuredTool
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
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


class ToolRetriever(BaseRetriever):
    """A retriever that searches for tools using RAG."""

    llm: ChatGroq = Field(default=None)
    vector_store: InMemoryVectorStore = Field(default=None)
    tool_registry: Dict[str, StructuredTool] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.llm is None:
            self.llm = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.0,
                streaming=False,
            )
        if self.vector_store is None:
            self.vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings())

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant tool documents based on query."""

        # Generate enhanced query using LLM
        enhanced_query = self._generate_enhanced_query(query)

        # Search vector store
        tool_documents = self.vector_store.similarity_search(enhanced_query, k=5)

        # Enhance documents with tool metadata
        enhanced_docs = []
        for doc in tool_documents:
            tool_name = doc.metadata.get("tool_name")
            if tool_name and tool_name in self.tool_registry:
                tool = self.tool_registry[tool_name]

                # Create enhanced document with tool information
                enhanced_doc = Document(
                    page_content=f"Tool: {tool.name}\nDescription: {tool.description}",
                    metadata={
                        "tool_name": tool.name,
                        "tool_type": type(tool).__name__,
                        "tool_object": tool,  # Include the actual tool object
                        "relevance_score": doc.metadata.get("relevance_score", 0.0),
                    },
                )
                enhanced_docs.append(enhanced_doc)

        return enhanced_docs

    def _generate_enhanced_query(self, original_query: str) -> str:
        """Generate an enhanced query using LLM for better tool retrieval."""

        system_prompt = SystemMessage(
            "Given this conversation, generate a query for additional tools. "
            "The query should be a short string containing what type of information "
            "is needed. Focus on tool capabilities and functionality."
        )

        human_message = HumanMessage(
            f"""
            From this query, could you give me a list of tools that can help with this query?
            Query: {original_query}
            """
        )

        try:
            response = self.llm.bind_tools([QueryForTools], tool_choice=True).invoke(
                [system_prompt, human_message]
            )
            enhanced_query = response.tool_calls[0]["args"]["query"]
            return enhanced_query
        except Exception:
            # Fallback to original query if LLM enhancement fails
            return original_query

    def add_tool(self, tool: StructuredTool):
        """Add a tool to the RAG system."""
        tool_id = str(uuid.uuid4())
        self.tool_registry[tool.name] = tool  # Use tool name as key for easier lookup

        tool_document = Document(
            page_content=f"{tool.name}: {tool.description}",
            metadata={
                "tool_id": tool_id,
                "tool_name": tool.name,
                "tool_type": type(tool).__name__,
            },
        )
        self.vector_store.add_documents([tool_document])

    def get_tool(self, tool_name: str) -> StructuredTool:
        """Retrieve a tool by name."""
        if tool_name in self.tool_registry:
            return self.tool_registry[tool_name]
        raise ValueError(f"Tool '{tool_name}' not found in RAG system.")

    def get_tools_by_query(self, query: str) -> List[StructuredTool]:
        """Search for tools based on a query and return tool objects."""
        documents = self._get_relevant_documents(query, run_manager=None)

        tools = []
        for doc in documents:
            tool_obj = doc.metadata.get("tool_object")
            if tool_obj:
                tools.append(tool_obj)

        return tools

    def search_tools(self, query: str, k: int = 5) -> List[Dict]:
        """Search for tools and return detailed information."""
        documents = self._get_relevant_documents(query, run_manager=None)

        results = []
        for doc in documents[:k]:
            result = {
                "tool_name": doc.metadata.get("tool_name"),
                "tool_type": doc.metadata.get("tool_type"),
                "description": doc.page_content,
                "relevance_score": doc.metadata.get("relevance_score", 0.0),
                "tool_object": doc.metadata.get("tool_object"),
            }
            results.append(result)

        return results


# Tool functions (same as before)
def cv_file_reader(file_path: str) -> str:
    """Read CV files in various formats (PDF, DOCX, TXT)"""
    try:
        # Simplified implementation (would use real parsers in production)
        return f"Successfully read CV from {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


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


def email_sender(recipient: str, subject: str, body: str) -> str:
    """Send email with CV analysis results"""
    # Simplified implementation
    return f"Email sent to {recipient} with subject '{subject}'"


# Create StructuredTools (same as before)
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


# Initialize and populate ToolRetriever
def setup_tool_retriever() -> ToolRetriever:
    """Setup and populate the ToolRRetriever."""
    tool_retriever = ToolRetriever()

    # Add tools
    tools = [
        cv_reader_tool,
        skill_extractor_tool,
        cv_comparison_tool,
        report_tool,
        email_tool,
        WriterAgent(),
        ComparatorAgent(),
        ScorecardAgent(),
    ]

    for tool in tools:
        tool_retriever.add_tool(tool)
        print(f"Added tool: {tool.name}")

    return tool_retriever

tool_retriever = setup_tool_retriever()