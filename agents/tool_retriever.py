from typing import Dict, List
from dotenv import load_dotenv
import uuid
import os

from langchain.tools import StructuredTool, tool
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
            self.vector_store = InMemoryVectorStore(
                embedding=OpenAIEmbeddings(
                    model="text-embedding-3-large",
                )
            )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant tool documents based on query."""

        # Generate enhanced query using LLM
        enhanced_query = self._generate_enhanced_query(query)

        # Search vector store
        tool_documents = self.vector_store.similarity_search(enhanced_query, k=1)

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


class ReadTxtFileInput(BaseModel):
    file_path: str = Field(
        ...,
        description="Extract the path of .txt file containing the context we need.",
    )


def read_txt_file(file_path: str) -> str:
    """Read a text file with automatic encoding detection and fixing."""
    try:
        from agents.utils import TextProcessor

        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."

        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a file."

        # Read file with error handling for encoding issues
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding if UTF-8 fails
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read().strip()

        if not content:
            return f"File '{file_path}' is empty."

        # Automatically detect and fix encoding issues
        original_length = len(content)
        fixed_content = TextProcessor.detect_and_fix_encoding(content)

        # Log if encoding fixes were applied
        if len(fixed_content) != original_length or "Ã" in content:
            print(f"🔧 Applied encoding fixes to '{file_path}'")
            print(
                f"   Original: {original_length} chars → Fixed: {len(fixed_content)} chars"
            )

        return fixed_content

    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


read_text_file_tool = StructuredTool.from_function(
    func=read_txt_file,
    name="read_txt_file",
    description="Read a file and return its contents with automatic encoding fix for special characters.",
    args_schema=ReadTxtFileInput,
)


class ReadTxtDirectoryInput(BaseModel):
    directory: str = Field(
        ...,
        description="Extract the directory path containing .txt files to read.",
    )


@tool(
    name_or_callable="read_txt_directory",
    description="Gather all txt files in a directory and return their contents in a single string.",
    args_schema=ReadTxtDirectoryInput,
)
def read_txt_directory(directory: str) -> str:
    """Read all .txt files from a directory and return their contents."""
    try:
        if not os.path.exists(directory):
            return f"Error: Directory '{directory}' does not exist."

        if not os.path.isdir(directory):
            return f"Error: '{directory}' is not a directory."

        file_contents = []
        # Get all .txt files
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                try:
                    file_contents.append(read_txt_file(file_path))
                except Exception as e:
                    file_contents.append(f"Error reading file: {e}")

        result = "\n\n".join(file_contents)

        return result

    except Exception as e:
        return f"Error accessing directory '{directory}': {str(e)}"


# Initialize and populate ToolRetriever
def setup_tool_retriever() -> ToolRetriever:
    """Setup and populate the ToolRetriever."""
    tool_retriever = ToolRetriever()

    # Add tools
    tools = [
        read_text_file_tool,
        read_txt_directory,
        WriterAgent(),
        ComparatorAgent(),
        ScorecardAgent(),
    ]

    for tool in tools:
        tool_retriever.add_tool(tool)

    return tool_retriever


tool_retriever = setup_tool_retriever()
