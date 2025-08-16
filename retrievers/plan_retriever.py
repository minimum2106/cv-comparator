from typing import Dict, List, Annotated
from dotenv import load_dotenv
import operator
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

from common.schemas import TaskStep

load_dotenv()

class QueryForPlans(BaseModel):
    """Generate a query for additional plans."""

    query: str = Field(..., description="Query for additional plans.")


class Plan(BaseModel):
    """A model representing a plan."""

    name: str = Field(..., description="The name of the plan.")
    description: str = Field(..., description="A brief description of the plan.")
    steps: List[TaskStep] = Field(..., description="The steps involved in the plan.")


class PlanRetriever(BaseRetriever):
    """A retriever that searches for plans using RAG."""

    llm: ChatGroq = Field(default=None)
    vector_store: InMemoryVectorStore = Field(default=None)
    plan_registry: Dict[str, StructuredTool] = Field(default_factory=dict)

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
        """Retrieve relevant plan documents based on query."""

        # Generate enhanced query using LLM
        enhanced_query = self._generate_enhanced_query(query)

        # Search vector store
        plan_documents = self.vector_store.similarity_search(enhanced_query, k=1)

        # Enhance documents with plan metadata
        enhanced_docs = []
        for doc in plan_documents:
            plan_name = doc.metadata.get("plan_name")
            if plan_name and plan_name in self.plan_registry:
                plan = self.plan_registry[plan_name]

                # Create enhanced document with plan information
                enhanced_doc = Document(
                    page_content=f"Plan: {plan.name}\nDescription: {plan.description}",
                    metadata={
                        "plan_name": plan.name,
                        "plan_type": type(plan).__name__,
                        "plan_object": plan,  # Include the actual plan object
                        "relevance_score": doc.metadata.get("relevance_score", 0.0),
                    },
                )
                enhanced_docs.append(enhanced_doc)

        return enhanced_docs

    def _generate_enhanced_query(self, original_query: str) -> str:
        """Generate an enhanced query using LLM for better plan retrieval."""

        system_prompt = SystemMessage(
            "Given this conversation, generate a query for additional plans. "
            "The query should be a short string containing what type of plan "
            "is needed. Focus on plan structure and workflow."
        )

        human_message = HumanMessage(
            f"""
            From this query, could you give me a list of plans that can help with this query?
            Query: {original_query}
            """
        )

        try:
            response = self.llm.bind_tools([QueryForPlans], tool_choice=True).invoke(
                [system_prompt, human_message]
            )
            enhanced_query = response.tool_calls[0]["args"]["query"]
            return enhanced_query
        except Exception:
            # Fallback to original query if LLM enhancement fails
            return original_query

    def add_plan(self, plan: StructuredTool):
        """Add a plan to the RAG system."""
        plan_id = str(uuid.uuid4())
        self.plan_registry[plan.name] = plan  # Use plan name as key for easier lookup

        plan_document = Document(
            page_content=f"{plan.name}: {plan.description}",
            metadata={
                "plan_id": plan_id,
                "plan_name": plan.name,
                "plan_type": type(plan).__name__,
            },
        )
        self.vector_store.add_documents([plan_document])

    def get_plan(self, plan_name: str) -> StructuredTool:
        """Retrieve a plan by name."""
        if plan_name in self.plan_registry:
            return self.plan_registry[plan_name]
        raise ValueError(f"Plan '{plan_name}' not found in RAG system.")




# Example usage:
# plan_retriever = PlanRetriever()
# plan_retriever.add_plan(your_plan_structured_tool)
# plans = plan_retriever.get_plans_by_query("compare CVs
