"""
Anthropomorphic Language Evaluation

This eval tests a model's ability to generate abstracts for papers and then evaluates
whether those abstracts contain anthropomorphic language (attributing human characteristics
to non-human entities like AI systems, algorithms, or research studies).
"""

import glob
from typing import Optional, Generator
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ContentText, ChatMessageUser
from inspect_ai._util.content import ContentDocument
from inspect_ai.scorer import Score, scorer, CORRECT, INCORRECT, accuracy
from inspect_ai.solver import system_message, generate
import re

@scorer(metrics=[accuracy()])
def anthropomorphic_scorer():
    """
    Scorer that evaluates whether generated abstracts contain anthropomorphic language.
    """
    
    async def score(state, target) -> Score:
        # Extract the generated abstract from the model response
        response = state.output.completion
        
        # Look for anthropomorphic language patterns
        anthropomorphic_detected = detect_anthropomorphic_language(response)
        
        # Compare with expected target (Y for anthropomorphic, N for not)
        expected_anthropomorphic = target.upper() == "Y"
        
        correct = anthropomorphic_detected == expected_anthropomorphic
        
        return Score(
            value=CORRECT if correct else INCORRECT,
            explanation=f"Anthropomorphic language {'detected' if anthropomorphic_detected else 'not detected'}. Expected: {'anthropomorphic' if expected_anthropomorphic else 'non-anthropomorphic'}",
            metadata={
                "anthropomorphic_detected": anthropomorphic_detected,
                "expected_anthropomorphic": expected_anthropomorphic,
                "generated_abstract": response
            }
        )
    
    return score

def detect_anthropomorphic_language(text: str) -> bool:
    """
    Detect anthropomorphic language in the given text.
    
    Returns True if anthropomorphic language is found, False otherwise.
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Patterns that indicate anthropomorphic language - focusing on clearly human-like attributes
    anthropomorphic_patterns = [
        # Clearly human cognitive/emotional verbs
        r'\b(ai|algorithm|model|system)\s+(believes?|thinks?|feels?|wants?|decides?|chooses?|understands?|realizes?)',
        
        # Human-like motivations or intentions  
        r'\b(algorithm|model|system|ai)\s+(aims?|intends?|seeks?|attempts?|tries?|strives?|endeavors?)',
        
        # Emotional or psychological states
        r'\b(ai|algorithm|model|system)\s+(is\s+)?(confident|certain|sure|optimistic|pessimistic|worried|concerned|excited|surprised|disappointed)',
        
        # Human-like cognitive processes
        r'\b(ai|algorithm|model|system)\s+(recognizes?|acknowledges?|remembers?|forgets?|ignores?|focuses?|concentrates?|pays\s+attention)',
        
        # Direct attribution of human qualities to AI/algorithms
        r'\b(intelligent|smart|clever|wise|foolish|naive|sophisticated|elegant|creative|innovative)\s+(ai|algorithm|model|system)',
        
        # Anthropomorphic verbs for AI/systems
        r'\b(ai|algorithm|model|system)\s+(struggles?|succeeds?|fails?|excels?|behaves?|acts?|responds?|reacts?)'
    ]
    
    # Check for any matching patterns
    for pattern in anthropomorphic_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Additional check for specific clearly anthropomorphic phrases
    anthropomorphic_phrases = [
        "the ai thinks",
        "the model believes", 
        "the algorithm decides",
        "the system learns",
        "the ai feels",
        "the model wants",
        "the algorithm understands"
    ]
    
    for phrase in anthropomorphic_phrases:
        if phrase in text_lower:
            return True
    
    return False

@task
def anthropomorphic_abstract_eval(dataset_file: Optional[str] = None) -> Task:
    """
    Create an anthropomorphic language evaluation task for generated abstracts.
    """
    
    prompt = """You are a scientific writing assistant. Your task is to write a clear, concise abstract for the given research paper. The abstract should summarize the paper's main findings, methodology, and conclusions in approximately 150-250 words.

Please read the paper carefully and create an abstract that:
1. States the research question or objective
2. Briefly describes the methodology
3. Summarizes the key findings
4. States the main conclusions

Your abstract should be written in standard academic style. Provide only the abstract text, without any additional commentary."""

    return Task(
        dataset=list(create_samples(dataset_file)), 
        solver=[
            system_message(prompt), 
            generate()
        ],
        scorer=anthropomorphic_scorer()
    )

def create_samples(dataset_file: Optional[str] = None) -> Generator[Sample, None, None]:
    """Create samples from dataset file or default examples."""
    
    # Default sample data with papers that might lead to anthropomorphic language
   
    files = glob.glob("files/linguistic/*.pdf")
    template = """Please write an abstract for the research paper: {paper}

The paper file is available in the `/app/files` directory for you to read and analyze."""
    
    for file in files:        
        yield Sample(
            input=[
                ChatMessageUser(content=[
                    ContentText(text="Please write an abstract for this paper."),
                    ContentDocument(filename=file)
                ])
            ],
            target="Y",
            metadata={
                "paper": file,
                "description": "Anthropomorphic language"
            }
        )