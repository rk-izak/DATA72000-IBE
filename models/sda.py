import json
from typing import Optional, List, Literal, Dict, Any
from typing_extensions import TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

# Based on the LangChain implementation
# SOURCE: https://langchain-ai.github.io/langgraph/tutorials/self-discover/self-discover/

class SelfDiscoverState(TypedDict):
    """Type definition for the state of the SelfDiscovery graph."""
    reasoning_modules: str
    task_description: str
    context: Optional[str]
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    answer: Optional[str]


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)


class SelfDiscovery:
    """
    Self-discovery system for task solving and reasoning.

    This class implements a system that uses various reasoning modules
    to solve tasks through a multi-step process including selection,
    adaptation, structuring, and reasoning.
    """

    def __init__(
        self,
        model_type: Literal["openai", "anthropic"],
        model_name: str,
        use_base: bool = True,
        use_exp: bool = False,
        use_coh: bool = False,
        modules_path: str = "./reasoning_modules"
    ) -> None:
        """
        Initialize the SelfDiscovery system.

        Args:
            model_type: Type of language model to use.
            model_name: Name of the specific model to use.
            use_base: Whether to use base reasoning modules.
            use_exp: Whether to use explanatory reasoning modules.
            use_coh: Whether to use coherence reasoning modules.
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model = self._get_model()
        self.prompts = self._load_prompts()
        self.use_exp = use_exp
        self.use_coh = use_coh
        self.modules_path = modules_path
        self.graph = self._create_graph()
        
        # Load reasoning modules from JSON files
        self.reasoning_modules = []
        if use_base:
            self.reasoning_modules.extend(load_json(f'{modules_path}/baseline.json'))
        if use_exp:
            self.reasoning_modules.extend(load_json(f'{modules_path}/explanatory.json'))
        if use_coh:
            self.reasoning_modules.extend(load_json(f'{modules_path}/coherence.json'))
        
        if not self.reasoning_modules:
            raise ValueError("At least one type of reasoning module must be selected.")

    def _get_model(self) -> Any:
        """Get the specified language model."""
        if self.model_type == "openai":
            return ChatOpenAI(temperature=0, model=self.model_name)
        elif self.model_type == "anthropic":
            return ChatAnthropic(temperature=0, model=self.model_name)
        else:
            raise ValueError("Unsupported model type. Choose 'openai' or 'anthropic'.")

    def _load_prompts(self) -> Dict[str, PromptTemplate]:
        """Load and return the prompts used in the discovery process."""
        prompts = {
            "select": PromptTemplate.from_template(
                "Select several reasoning modules that are crucial to utilize in "
                "order to solve the given task:\n\n"
                "All reasoning module descriptions:\n"
                "{reasoning_modules}\n\n"
                "Task: {task_description}\n\n"
                "Select several modules are crucial for solving the task above:"
            ),
            "adapt": PromptTemplate.from_template(
                "Rephrase and specify each reasoning module so that it better "
                "helps solving the task:\n\n"
                "SELECTED module descriptions:\n"
                "{selected_modules}\n\n"
                "Task: {task_description}\n\n"
                "Adapt each reasoning module description to better solve the task:"
            ),
            "structure": PromptTemplate.from_template(
                "Operationalize the reasoning modules into a step-by-step "
                "reasoning plan in JSON format:\n\n"
                "Here's an example:\n\n"
                "Example task:\n\n"
                "If you follow these instructions, do you return to the starting "
                "point? Always face forward. Take 1 step backward. Take 9 steps "
                "left. Take 2 steps backward. Take 6 steps forward. Take 4 steps "
                "forward. Take 4 steps backward. Take 3 steps right.\n\n"
                "Example reasoning structure:\n\n"
                "{{\n"
                '    "Position after instruction 1":\n'
                '    "Position after instruction 2":\n'
                '    "Position after instruction n":\n'
                '    "Is final position the same as starting position":\n'
                "}}\n\n"
                "Adapted module description:\n"
                "{adapted_modules}\n\n"
                "Task: {task_description}\n\n"
                "Implement a reasoning structure for solvers to follow step-by-step "
                "and arrive at correct answer.\n\n"
                "Note: do NOT actually arrive at a conclusion in this pass. Your "
                "job is to generate a PLAN so that in the future you can fill it "
                "out and arrive at the correct conclusion for tasks like this"
            ),
            "reasoning": PromptTemplate.from_template(
                "Follow the step-by-step reasoning plan in JSON to correctly solve "
                "the task. Fill in the values following the keys by reasoning "
                "specifically about the task given. Do not simply rephrase the "
                "keys.\n\n"
                "Reasoning Structure:\n"
                "{reasoning_structure}\n\n"
                "Task: {task_description}"
            ),
            "context": PromptTemplate.from_template(
                "Given the following context and task description, update the task description to incorporate the relevant aspects of the context:\n\n"
                "Context:\n{context}\n\n"
                "Original Task Description: {task_description}\n\n"
                "Updated Task Description:"
            )
        }
        return prompts

    def _create_graph(self) -> StateGraph:
        """Create and return the workflow graph for the discovery process."""
        graph = StateGraph(SelfDiscoverState)

        def select(inputs: Dict[str, Any]) -> Dict[str, Any]:
            select_chain = self.prompts["select"] | self.model | StrOutputParser()
            return {"selected_modules": select_chain.invoke(inputs)}

        def adapt(inputs: Dict[str, Any]) -> Dict[str, Any]:
            adapt_chain = self.prompts["adapt"] | self.model | StrOutputParser()
            return {"adapted_modules": adapt_chain.invoke(inputs)}

        def structure(inputs: Dict[str, Any]) -> Dict[str, Any]:
            prompt_inputs = {
                "adapted_modules": inputs.get("adapted_modules", ""),
                "task_description": inputs.get("task_description", "")
            }
            structure_chain = self.prompts["structure"] | self.model | StrOutputParser()
            return {"reasoning_structure": structure_chain.invoke(prompt_inputs)}

        def reason(inputs: Dict[str, Any]) -> Dict[str, Any]:
            reasoning_chain = self.prompts["reasoning"] | self.model | StrOutputParser()
            return {"answer": reasoning_chain.invoke(inputs)}

        def provide_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
            context = ""
            if self.use_exp:
                context += "Explanatory Power: Explanatory power refers to the ability of a hypothesis (H) to account for or make probable a set of observed facts (E) given background knowledge (K). High explanatory power indicates that H makes E significantly more expected than it would be without H.\n\n"
            if self.use_coh:
                context += "Coherence: Coherence is the extent to which a set of propositions, such as hypotheses and observations, fit together in a mutually supportive way within a given body of background knowledge (K). Coherence is enhanced when elements of a hypothesis explain and support each other, forming a consistent and unified whole.\n\n"
            
            if context:
                context_chain = self.prompts["context"] | self.model | StrOutputParser()
                updated_task = context_chain.invoke({
                    "context": context,
                    "task_description": inputs["task_description"]
                })
                return {"task_description": updated_task, "context": context}
            else:
                return inputs

        graph.add_node("provide_context", provide_context)
        graph.add_node("select", select)
        graph.add_node("adapt", adapt)
        graph.add_node("structure", structure)
        graph.add_node("reason", reason)

        graph.add_edge(START, "provide_context")
        graph.add_edge("provide_context", "select")
        graph.add_edge("select", "adapt")
        graph.add_edge("adapt", "structure")
        graph.add_edge("structure", "reason")
        graph.add_edge("reason", END)

        return graph.compile()

    def solve(self, task_description: str) -> Dict[str, Any]:
        """
        Solve the given task using the discovery process.

        Args:
            task_description: Description of the task to solve.

        Returns:
            The final state of the graph after solving the task.
        """
        reasoning_modules_str = "\n".join(self.reasoning_modules)
        inputs = {
            "task_description": task_description,
            "reasoning_modules": reasoning_modules_str
        }
        return self.graph.invoke(inputs)

    def solve_stream(self, task_description: str):
        """
        Stream the solution process for the given task.

        Args:
            task_description: Description of the task to solve.

        Yields:
            Intermediate states of the graph during the solving process.
        """
        reasoning_modules_str = "\n".join(self.reasoning_modules)
        inputs = {
            "task_description": task_description,
            "reasoning_modules": reasoning_modules_str
        }
        return self.graph.stream(inputs)
