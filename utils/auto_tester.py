import json
import re
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class AutomaticTester:
    """
    A class for automatically testing and evaluating language models on various tasks.
    """

    def __init__(self, sda_agent: Any, rag_agent: Any, input_filepath: str, output_filepath: str):
        """
        Initialize the AutomaticTester.

        Args:
            sda_agent: The agent for solving tasks.
            rag_agent: The agent for retrieving relevant evidence.
            input_filepath: Path to the input JSON file.
            output_filepath: Path to save the output results.
        """
        self.sda_agent = sda_agent
        self.rag_agent = rag_agent
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        print(f"\n--- INITIALIZING AUTOMATIC TESTER ---")
        print(f"Input filepath: {input_filepath}")
        print(f"Output filepath: {output_filepath}")

    def format_entries(self, task: str, task_type: str) -> List[str]:
        """
        Format entries from the input file based on the task type.

        Args:
            task: The task description.
            task_type: The type of task (explanation, selection, or assignment).

        Returns:
            A list of formatted entries.
        """
        print(f"\n--- FORMATTING ENTRIES ---")
        print(f"Task Type: {task_type}")
        try:
            with open(self.input_filepath, 'r') as file:
                data = json.load(file)
            print(f"Successfully loaded {len(data)} entries from {self.input_filepath}")

            formatted_entries = []

            for i, item in enumerate(data, 1):
                print(f"\nFormatting entry {i}/{len(data)}")
                output = f"Task: {task}:\n\n"

                if task_type == "explanation":
                    output += self._format_explanation_entry(item)
                else:  # "selection" or "assignment" options
                    output += self._format_selection_assignment_entry(item, task_type)

                formatted_entries.append(output)

            print(f"Successfully formatted {len(formatted_entries)} entries")
            return formatted_entries

        except Exception as e:
            print(f"An error occurred while formatting entries: {str(e)}")
            return []

    def _format_explanation_entry(self, item: Dict[str, Any]) -> str:
        """Helper method to format an explanation entry."""
        output = f"Claim: {item['claim']}\n\n"
        output += "Evidence List:\n"

        for evidence in item['evidence']:
            output += f"- ID {evidence['evidence_id']}: {evidence['description']}\n"

        if 'wrong_evidence' in item:
            for wrong_evidence in item['wrong_evidence']:
                output += f"- ID {wrong_evidence['evidence_id']}: {wrong_evidence['description']}\n"

        output += "\nAdditional Information:\n"
        if 'context' in item:
            for key, value in item['context'].items():
                output += f"- {key}: {value if value else 'Not Found'}\n"
        else:
            output += "  No additional information available.\n"

        return output

    def _format_selection_assignment_entry(self, item: Dict[str, Any], task_type: str) -> str:
        """Helper method to format a selection or assignment entry."""
        output = f"Claim A: {item['claim_A']}\n\n"
        output += "Additional Information:\n"
        if 'context_A' in item:
            for key, value in item['context_A'].items():
                output += f"- {key}: {value if value else 'Not Found'}\n"
        else:
            output += "  No additional information available.\n"

        output += f"\nClaim B: {item['claim_B']}\n\n"
        output += "Additional Information:\n"
        if 'context_B' in item:
            for key, value in item['context_B'].items():
                output += f"- {key}: {value if value else 'Not Found'}\n"
        else:
            output += "  No additional information available.\n"

        evidence_list = []
        if task_type == "selection":
            evidence_list.extend(item.get('evidence_A', []))
            evidence_list.extend(item.get('evidence_B', []))
        elif task_type == "assignment":
            evidence_list.extend(item.get('evidence_A', []))
            evidence_list.extend(item.get('evidence_B', []))
            evidence_list.extend(item.get('evidence_C', []))

        output += "\nEvidence List:\n"
        for evidence in evidence_list:
            output += f"- ID {evidence['evidence_id']}: {evidence['description']}\n"

        return output

    def generate_questions(self, task: str, n: int = 3) -> str:
        """
        Generate questions for the given task.

        Args:
            task: The task description.
            N: Number of questions to generate.

        Returns:
            A string containing the question generation task.
        """
        print(f"\n--- GENERATING QUESTIONS ---")
        print(f"Number of questions to generate: {n}")
        q_task = (
            f"Generate {n} useful questions to solve this task:\n{task}\n"
            f"Focus on missing information, not solution methods. Ensure diversity of information and avoid repetition. Format as:\n"
            f"Question 1: ...\n"
            f"Question 2: ...\n"
            f"Continue until {n} questions are generated."
        )
        print("Question generation task created")
        return q_task

    def extract_questions(self, text: str, n: int = 1) -> Dict[str, str]:
        """
        Extract questions from the generated text.

        Args:
            text: The text containing the generated questions.
            n: Number of questions to extract.

        Returns:
            A dictionary of extracted questions.
        """
        print(f"\n--- EXTRACTING QUESTIONS ---")
        print(f"Extracting {n} questions from the generated text")
        llm = ChatOpenAI(model="gpt-4o-mini")
        prompt = ChatPromptTemplate.from_template(
            """Extract {n} question(s) from the following text. Do not paraphrase or in any other way change them. Simply extract them.
            These should be found at the end of the provided text with the word "Question" before each.
            Format your response as a JSON object where the keys are "Question 1", "Question 2", etc., and the values are the extracted questions.
            If you cannot find {n} question(s), extract however many there might be. DO NOT generate new questions.

            Text:
            {text}

            Output your response in the following JSON format:
            {{
                "Question 1": "...",
                "Question 2": "...",
                ...
            }}
            """
        )

        chain = prompt | llm
        response = chain.invoke({"text": text, "n": n})
        content = response.content

        try:
            questions_dict = json.loads(content)
            print("Successfully extracted questions as JSON")
        except json.JSONDecodeError:
            print("Failed to parse JSON, falling back to regex extraction")
            questions_dict = {}
            pattern = r'"Question (\d+)":\s*"([^"]+)"'
            matches = re.findall(pattern, content)
            for num, question in matches:
                questions_dict[f"question_{num}"] = question

        result = {f"question_{i+1}": v for i, (k, v) in enumerate(questions_dict.items())}
        print(f"Extracted {len(result)} questions")
        return result

    def extract_explanation_evidence(self, text: str, task_type: str = "explanation") -> Dict[str, Any]:
        """
        Extract explanation and evidence from the generated text.

        Args:
            text: The text containing the explanation and evidence.
            task_type: The type of task (explanation, assignment, or selection).

        Returns:
            A dictionary containing the extracted explanation and evidence.
        """
        print(f"\n--- EXTRACTING EXPLANATION AND EVIDENCE ---")
        print(f"Task Type: {task_type}")
        llm = ChatOpenAI(model="gpt-4o-mini")

        prompt_template = self._get_prompt_template(task_type)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = chain.invoke({"text": text})
        content = response.content

        print("Extraction complete. Parsing results...")

        lines = content.split('\n')
        result = {}

        if task_type == "explanation":
            result = self._parse_explanation_result(lines)
        elif task_type == "assignment":
            result = self._parse_assignment_result(lines)
        elif task_type == "selection":
            result = self._parse_selection_result(lines)
        else:
            raise ValueError("Invalid task_type. Must be 'explanation', 'assignment', or 'selection'.")

        print("Parsing complete")
        return result

    def _get_prompt_template(self, task_type: str) -> str:
        """Helper method to get the appropriate prompt template based on task type."""
        if task_type == "explanation":
            return """Extract the the generated explanation from the following text.
            This should be found at the end of the provided text, however, this might not always be the case.
            If you can't find it, return an empty string for the explanation. DO NOT generate anything not found in text.

            Text:
            {text}

            Output your response in the following format:
            Generated Explanation: string
            """
        elif task_type == "assignment":
            return """Extract the generated explanations for Claim A and Claim B from the following text.
            These should be found at the end of the provided text, however, this might not always be the case.
            If you can't find any of these return empty strings for explanations. DO NOT generate anything not found in text.

            Text:
            {text}

            Output your response in the following format:
            Claim A Generated Explanation: string
            Claim B Generated Explanation: string
            """
        elif task_type == "selection":
            return """Extract the selected claim (A or B) and the generated explanation from the following text.
            These should be found at the end of the provided text, however, this might not always be the case.
            If you can't find any of these, return an empty string for the selected claim or an empty string for the explanation. DO NOT generate anything not found in text.

            Text:
            {text}

            Output your response in the following format:
            Selected Claim: string
            Generated Explanation: string
            """
        else:
            raise ValueError("Invalid task_type. Must be 'explanation', 'assignment', or 'selection'.")

    def _parse_explanation_result(self, lines: List[str]) -> Dict[str, Any]:
        """Helper method to parse the result for explanation task type."""
        # ids_line = next((line for line in lines if line.startswith('Evidence IDs:')), '')
        explanation_line = next((line for line in lines if line.startswith('Generated Explanation:')), '')

        # ids_string = ids_line.replace('Evidence IDs:', '').strip()
        # ids_list = eval(ids_string) if ids_string else []

        explanation = explanation_line.replace('Generated Explanation:', '').strip()

        return {
            # "selected_ids": ids_list,
            "generated_explanation": explanation
        }

    def _parse_assignment_result(self, lines: List[str]) -> Dict[str, Any]:
        """Helper method to parse the result for assignment task type."""
        # claim_a_ids_line = next((line for line in lines if line.startswith('Claim A Evidence IDs:')), '')
        claim_a_explanation_line = next((line for line in lines if line.startswith('Claim A Generated Explanation:')), '')
        # claim_b_ids_line = next((line for line in lines if line.startswith('Claim B Evidence IDs:')), '')
        claim_b_explanation_line = next((line for line in lines if line.startswith('Claim B Generated Explanation:')), '')

        # claim_a_ids_string = claim_a_ids_line.replace('Claim A Evidence IDs:', '').strip()
        # claim_a_ids_list = eval(claim_a_ids_string) if claim_a_ids_string else []
        claim_a_explanation = claim_a_explanation_line.replace('Claim A Generated Explanation:', '').strip()

        # claim_b_ids_string = claim_b_ids_line.replace('Claim B Evidence IDs:', '').strip()
        # claim_b_ids_list = eval(claim_b_ids_string) if claim_b_ids_string else []
        claim_b_explanation = claim_b_explanation_line.replace('Claim B Generated Explanation:', '').strip()

        return {
            "claim_A": {
                # "selected_ids": claim_a_ids_list,
                "generated_explanation": claim_a_explanation
            },
            "claim_B": {
                # "selected_ids": claim_b_ids_list,
                "generated_explanation": claim_b_explanation
            }
        }

    def _parse_selection_result(self, lines: List[str]) -> Dict[str, Any]:
        """Helper method to parse the result for selection task type."""
        selected_claim_line = next((line for line in lines if line.startswith('Selected Claim:')), '')
        # ids_line = next((line for line in lines if line.startswith('Evidence IDs:')), '')
        explanation_line = next((line for line in lines if line.startswith('Generated Explanation:')), '')

        selected_claim = selected_claim_line.replace('Selected Claim:', '').strip()
        # ids_string = ids_line.replace('Evidence IDs:', '').strip()
        # ids_list = eval(ids_string) if ids_string else []
        explanation = explanation_line.replace('Generated Explanation:', '').strip()

        return {
            "selected_claim": selected_claim,
            # "selected_ids": ids_list,
            "generated_explanation": explanation
        }

    def run_test(self, task_type: str, use_rag: bool = False, num_examples: int = -1) -> None:
            """
            Run the test based on the specified parameters.

            Args:
                task_type: The type of task (explanation, assignment, or selection).
                use_rag: Whether to use RAG (Retrieval-Augmented Generation) or not.
                num_examples: Number of examples to process (-1 for all).
            """
            task_prompts = {
                "explanation": "Generate a 3-5 sentence explanation of the claim using relevant evidence. "
                            "Critically address how evidence supports or contradicts the claim. "
                            "Be cautious of incomplete or incorrect evidence. "
                            "Incorporate additional information if provided and useful. "
                            "Format output as:\n"
                            "- Generated Explanation: [explanation]",

                "assignment": "Generate 3-5 sentence explanations for claims A and B using relevant evidence. "
                            "Identify supporting or contradicting evidence for each claim. "
                            "Be cautious of unrelated, incomplete, or incorrect evidence. "
                            "Incorporate additional context if provided and useful. "
                            "Format output as:\n"
                            "- Claim A:\n"
                            "  - Generated Explanation: [explanation for A]\n\n"
                            "- Claim B:\n"
                            "  - Generated Explanation: [explanation for B]",

                "selection": "Select appropriate claim (A or B) based on provided evidence. "
                            "Generate a 3-5 sentence explanation of chosen claim using evidence. "
                            "Critically address how evidence supports or contradicts the claim. "
                            "Be cautious of incomplete or incorrect evidence. "
                            "Incorporate additional context if provided and useful. "
                            "Format output as:\n"
                            "- Selected Claim: [A or B]\n"
                            "- Generated Explanation: [explanation]"
            }

            print("\n--- STARTING RUN_TEST ---")
            print(f"Task Type: {task_type}")
            print(f"Use RAG: {use_rag}")
            print(f"Number of Examples: {num_examples if num_examples > 0 else 'All'}")

            formatted_entries = self.format_entries(task_prompts[task_type], task_type)
            if num_examples > 0:
                formatted_entries = formatted_entries[:num_examples]
            print(f"Total entries to process: {len(formatted_entries)}")

            results = []

            for i, entry in enumerate(formatted_entries, 1):
                print(f"\n--- PROCESSING ENTRY {i}/{len(formatted_entries)} ---")
                print("--- ORIGINAL TASK PROMPT ---")
                print(entry)

                # Extract existing evidence IDs
                existing_evidence_ids = self._extract_existing_evidence_ids(entry)
                print(f"\nTotal Evidence IDs in Original Prompt: {len(existing_evidence_ids)}")

                unique_rag_evidence = []  # Initialize here, outside the use_rag condition

                if use_rag:
                    unique_rag_evidence = self._process_rag(entry, existing_evidence_ids)
                    entry = self._update_entry_with_rag(entry, unique_rag_evidence)

                    print("\n--- UPDATED TASK PROMPT WITH RAG EVIDENCE ---")
                    print(entry)

                print(f"\nTotal Evidence IDs in Final Prompt: {len(existing_evidence_ids) + len(unique_rag_evidence)}")

                print("\n--- SOLVING TASK ---")
                result = self.sda_agent.solve(entry)
                print("Task solved. Extracting explanation and evidence...")
                extracted_result = self.extract_explanation_evidence(result['answer'], task_type)
                results.append(extracted_result)

                print("\n--- EXTRACTED RESULT ---")
                print(json.dumps(extracted_result, indent=2))

            print("\n--- SAVING RESULTS ---")
            with open(self.output_filepath, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Results saved to {self.output_filepath}")
            print("--- RUN_TEST COMPLETED ---")

    def _extract_existing_evidence_ids(self, entry: str) -> set:
        """Helper method to extract existing evidence IDs from the entry."""
        existing_evidence_ids = set()
        for line in entry.split('\n'):
            if line.startswith("- ID"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    id_part = parts[0].split("ID", 1)[1].strip()
                    try:
                        existing_evidence_ids.add(int(id_part))
                    except ValueError:
                        existing_evidence_ids.add(id_part)
        return existing_evidence_ids

    def _process_rag(self, entry: str, existing_evidence_ids: set) -> List[Dict[str, Any]]:
        """Helper method to process RAG and get unique evidence."""
        print("\n--- GENERATING QUESTIONS FOR RAG ---")
        q_task = self.generate_questions(entry, n=2)
        q_result = self.sda_agent.solve(q_task)
        questions = self.extract_questions(q_result['answer'], n=2)
        print("Generated Questions:")
        for q_num, q_text in questions.items():
            print(f"{q_num}: {q_text}")

        print("\n--- RETRIEVING RAG EVIDENCE ---")
        rag_evidence = []
        for q in questions.values():
            rag_evidence.extend(self.rag_agent.get_relevant_evidence(q))
        print(f"Total RAG evidence retrieved: {len(rag_evidence)}")

        # Filter and add unique RAG evidence
        unique_rag_evidence = []
        seen_rag_ids = set()
        for evidence in rag_evidence:
            if isinstance(evidence, dict) and 'description' in evidence and 'evidence_id' in evidence:
                evidence_id = evidence['evidence_id']
                try:
                    evidence_id = int(evidence_id)
                except ValueError:
                    pass
                if evidence_id not in existing_evidence_ids and evidence_id not in seen_rag_ids:
                    unique_rag_evidence.append(evidence)
                    seen_rag_ids.add(evidence_id)

        print(f"Number of Unique RAG Evidence Added: {len(unique_rag_evidence)}")
        return unique_rag_evidence

    def _update_entry_with_rag(self, entry: str, unique_rag_evidence: List[Dict[str, Any]]) -> str:
        """Helper method to update the entry with RAG evidence."""
        entry_lines = entry.split('\n')
        evidence_end = next(i for i in range(len(entry_lines)-1, -1, -1) if entry_lines[i].startswith("Evidence List:"))
        for evidence in unique_rag_evidence:
            entry_lines.insert(evidence_end + 1, f"- ID {evidence['evidence_id']}: {evidence['description']}")
            evidence_end += 1
        return '\n'.join(entry_lines)

    def run(self, task_type: str, use_rag: bool = False, num_examples: int = -1):
        """
        Run the automatic tester with the specified parameters.

        Args:
            task_type: The type of task (explanation, assignment, or selection).
            use_rag: Whether to use RAG (Retrieval-Augmented Generation) or not.
            num_examples: Number of examples to process (-1 for all).
        """
        print("\n=== STARTING AUTOMATIC TESTER ===")
        print(f"Task Type: {task_type}")
        print(f"Use RAG: {use_rag}")
        print(f"Number of Examples: {num_examples if num_examples > 0 else 'All'}")

        try:
            self.run_test(task_type, use_rag, num_examples)
            print("\n=== AUTOMATIC TESTER COMPLETED SUCCESSFULLY ===")
        except Exception as e:
            print(f"\n=== ERROR IN AUTOMATIC TESTER ===")
            print(f"An error occurred: {str(e)}")
        finally:
            print("=== AUTOMATIC TESTER FINISHED ===")
