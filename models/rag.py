import json
from typing import List, Dict, Union, Tuple
from typing_extensions import TypedDict

import numpy as np
from scipy import stats

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic


class GraphState(TypedDict):
    """Type definition for the state of the RAG graph."""
    question: str
    generation: str
    documents: List[Dict[str, Union[int, str]]]
    evidence: List[Dict[str, Union[int, str]]]
    transform_count: int


class GradeDocuments(BaseModel):
    """Model for grading document relevance."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
    reasoning: str = Field(description="Explanation for the relevance score")


class GradeHallucinations(BaseModel):
    """Model for grading answer hallucinations."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")
    reasoning: str = Field(description="Explanation for the hallucination score")


class GradeAnswer(BaseModel):
    """Model for grading answer quality."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")
    reasoning: str = Field(description="Explanation for the answer score")


class SelfRAG:
    """
    Self-reflective Retrieval-Augmented Generation system.

    This class implements a RAG system with self-reflection capabilities,
    including document retrieval, relevance grading, and answer generation.
    """

    def __init__(self, llm_model: str, embedding_model: str):
        """
        Initialize the SelfRAG system.

        Args:
            llm_model (str): Name of the language model to use.
            embedding_model (str): Name of the embedding model to use.
        """
        if llm_model.startswith("gpt"):
            self.llm = ChatOpenAI(model=llm_model, temperature=0)
        elif llm_model.startswith("claude"):
            self.llm = ChatAnthropic(model=llm_model, temperature=0)
        else:
            raise ValueError("Unsupported LLM model")

        try:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        except Exception as e:
            print(f"Embedding Model loading failed: {e}")
            print("Have you provided a valid name?")

        self.vectorstore = None
        self.workflow = self._create_workflow()
        self.evidence_data = {}

    def load_documents(self, json_file: str) -> None:
        """
        Load documents from a JSON file and create a vector store.

        Args:
            json_file (str): Path to the JSON file containing documents.
        """
        with open(json_file, 'r') as f:
            data = json.load(f)

        documents = []
        for item in data:
            doc = f"Evidence ID: {item['evidence_id']}\nEvidence Description: {item['description']}"
            documents.append(Document(page_content=doc, metadata={"evidence_id": item['evidence_id']}))
            self.evidence_data[item['evidence_id']] = item['description']

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=self.embeddings,
        )

    def _create_workflow(self) -> StateGraph:
        """
        Create the workflow for the RAG system.

        Returns:
            StateGraph: The compiled workflow graph.
        """
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
                "end": END,
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate", # hallucinations
                "useful": END, # if is useful
                "not useful": "transform_query", # if not hallucinations, is it useful
            },
        )

        return workflow.compile()

    def retrieve(self, state: GraphState) -> GraphState:
        """
        Retrieve relevant documents based on the question.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            GraphState: Updated state with retrieved documents.
        """
        question = state["question"]
        print(f"\n---RETRIEVE---\nQuestion: {question}")

        # Retrieve top 50 documents
        relevant_documents = self.vectorstore.similarity_search_with_relevance_scores(question, k=5)

        # SCORE-BASED FILTER - UNUSED
        # # Extract scores
        # scores = np.array([score for _, score in documents]) 

        # # Calculate statistical measures
        # mean_score = np.mean(scores)
        # std_score = np.std(scores)
        # z_scores = stats.zscore(scores)

        # # Define thresholds
        # score_threshold = mean_score + 0.8 * std_score
        # z_score_threshold = 1.4

        # # Filter documents based on statistical measures
        # relevant_documents = [
        #     doc for doc, score, z in zip(documents, scores, z_scores)
        #     if score > score_threshold or z > z_score_threshold
        # ]

        print(f"Retrieved {len(relevant_documents)} relevant documents")
        # print(f"Mean score: {mean_score:.4f}, Std: {std_score:.4f}")
        # print(f"Score threshold: {score_threshold:.4f}, Z-score threshold: {z_score_threshold:.4f}")

        return {**state, "documents": [doc for doc, _ in relevant_documents]}

    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Grade the relevance of retrieved documents.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            GraphState: Updated state with graded documents.
        """
        question = state["question"]
        documents = state["documents"]
        print("\n---GRADE DOCUMENTS---")

        grader = self.llm.with_structured_output(GradeDocuments)
        system = """You are an expert grader assessing the relevance of retrieved documents to a user question. 
            Your goal is to identify documents that contain information directly related to answering the question.
            Consider both explicit keyword matches and implicit semantic relevance.
            Provide a binary score 'yes' or 'no' and explain your reasoning."""
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User question: {question}\n\nRetrieved document:\n{document}\n\nIs this document relevant? Explain your reasoning."),
        ])
        retrieval_grader = grade_prompt | grader

        filtered_docs = []
        for d in documents:
            result = retrieval_grader.invoke({"question": question, "document": d.page_content})
            print(f"Document relevance: {result.binary_score}\nReasoning: {result.reasoning}")
            if result.binary_score == "yes":
                filtered_docs.append(d)

        print(f"Filtered to {len(filtered_docs)} relevant documents")
        return {**state, "documents": filtered_docs}

    def generate(self, state: GraphState) -> GraphState:
        """
        Generate an answer based on the question and relevant documents.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            GraphState: Updated state with generated answer.
        """
        question = state["question"]
        documents = state["documents"]
        print("\n---GENERATE---")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant tasked with answering questions based on the provided context. "
                       "Ensure your response is directly relevant to the question and grounded in the given information."
                       "If you don't know the answer, just say that you don't know." 
                       "Use five sentences maximum and keep the answer concise."),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nProvide a comprehensive answer:"),
        ])
        rag_chain = prompt | self.llm | StrOutputParser()

        context = "\n\n".join([doc.page_content for doc in documents])
        generation = rag_chain.invoke({"context": context, "question": question})
        print(f"Generated answer:\n{generation}")
        return {**state, "generation": generation}

    def transform_query(self, state: GraphState) -> GraphState:
        """
        Transform the original query to improve document retrieval.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            GraphState: Updated state with transformed query.
        """
        question = state["question"]
        documents = state["documents"]
        transform_count = state.get("transform_count", 0) + 1
        print(f"\n---TRANSFORM QUERY (Attempt {transform_count})---")

        system = """You are an expert at reformulating questions to improve information retrieval.
            Analyze the input question and create a version that is more likely to match relevant documents.
            Consider expanding abbreviations, including synonyms, and clarifying ambiguous terms."""
        re_write_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Original question: {question}\n\nProvide an improved version for document retrieval:"),
        ])

        question_rewriter = re_write_prompt | self.llm | StrOutputParser()
        better_question = question_rewriter.invoke({"question": question})
        print(f"Transformed question: {better_question}")
        return {**state, "question": better_question, "transform_count": transform_count}

    def decide_to_generate(self, state: GraphState) -> str:
        """
        Decide whether to generate an answer, transform the query, or end the process.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            str: Decision to generate, transform query, or end.
        """
        filtered_documents = state["documents"]
        transform_count = state.get("transform_count", 0)
        
        if filtered_documents:
            decision = "generate"
        elif transform_count < 1:
            decision = "transform_query"
        else:
            decision = "end"
        
        print(f"\n---DECISION: {decision.upper()}---")
        return decision

    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        """
        Grade the generated answer against documents and the original question.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            str: Decision on the quality of the generated answer.
        """
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        print("\n---GRADE GENERATION---")

        hallucination_grader = self.llm.with_structured_output(GradeHallucinations)
        system = """You are an expert fact-checker assessing whether an AI-generated answer is grounded in the provided documents.
            Carefully compare the answer to the information in the documents.
            Provide a binary score 'yes' or 'no' and explain your reasoning."""
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Documents:\n{documents}\n\nAI-generated answer: {generation}\n\nIs this answer grounded in the documents? Explain your reasoning."),
        ])
        hallucination_check = hallucination_prompt | hallucination_grader

        hallucination_result = hallucination_check.invoke({"documents": [doc.page_content for doc in documents], "generation": generation})
        print(f"Hallucination check: {hallucination_result.binary_score}\nReasoning: {hallucination_result.reasoning}")

        if hallucination_result.binary_score == "yes":
            answer_grader = self.llm.with_structured_output(GradeAnswer)
            system = """You are an expert evaluator assessing whether an answer fully addresses and resolves a given question.
                Consider completeness, relevance, and clarity of the answer.
                Provide a binary score 'yes' or 'no' and explain your reasoning."""
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", "Question: {question}\n\nAnswer: {generation}\n\nDoes this answer fully address the question? Explain your reasoning."),
            ])
            answer_check = answer_prompt | answer_grader

            answer_result = answer_check.invoke({"question": question, "generation": generation})
            print(f"Answer quality check: {answer_result.binary_score}\nReasoning: {answer_result.reasoning}")

            return "useful" if answer_result.binary_score == "yes" else "not useful"
        else:
            return "not supported"

    def get_relevant_evidence(self, question: str) -> List[Dict[str, Union[int, str]]]:
        """
        Get relevant evidence for a given question.

        Args:
            question (str): The input question.

        Returns:
            List[Dict[str, Union[int, str]]]: List of relevant evidence with IDs and descriptions.
        """
        inputs = {"question": question, "documents": [], "generation": "", "evidence": [], "transform_count": 0}

        for output in self.workflow.stream(inputs):
            if "end" in output:
                return []
            if "generate" in output:
                relevant_docs = output["generate"]["documents"]
                if not relevant_docs:
                    return []
                evidence = []
                for doc in relevant_docs:
                    evidence_id = doc.metadata["evidence_id"]
                    evidence.append({
                        "evidence_id": evidence_id,
                        "description": self.evidence_data[evidence_id]
                    })
                return evidence

        return []
