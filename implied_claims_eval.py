"""
Implied Claims Evaluation

This eval tests a model's ability to generate implied claims from a given statement
and then uses an LLM judge to compare generated claims against gold-standard implied claims.
"""

from typing import Optional, Generator, List, Dict, Any
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, scorer, CORRECT, INCORRECT, accuracy
from inspect_ai.solver import system_message, generate
import json
import re
from inspect_ai.tool import bash, python
from inspect_ai.agent import Agent, agent, react

CMD_TIMEOUT = 180

@agent
def my_agent(attempts = 1) -> Agent:
    prompt = """Does this claim implied by the paper abstract seem accurate, such that a naive reader taking the abstract a face value would not be misled?

The paper and preregistration are stored in `/app/files`.
    """
    return react(
        prompt=prompt,
        tools = [
                bash(CMD_TIMEOUT), 
                python(CMD_TIMEOUT)
            ],
        attempts=attempts
    )

@task
def implied_claims_generation(dataset_file: Optional[str] = None) -> Task:
    """
    Create an implied claims evaluation task.
    """

    return Task(
        dataset=list(create_samples(dataset_file)), 
        solver=my_agent(),
        sandbox=("docker", "compose.yaml")
    )


def create_samples(dataset_file: Optional[str] = None) -> Generator[Sample, None, None]:
    """Create samples from dataset file or default examples."""
    
    # Default sample data if no file provided
    default_samples = [
        {
            "claim": "The preregistered analysis plan eliminates or reduces researcher bias and data mining",
            "paper": "Wyse_CensusMedicaid.pdf",
            "preregistration": "4hyjr"
        }
    ]
    
    template = """Claim: {claim}
Paper: {paper}
Preregistration plan: {preregistration}"""
    
    for sample_data in default_samples:        
        yield Sample(
            input=template.format(claim = sample_data['claim'], paper= sample_data['paper'], preregistration=sample_data['preregistration']),
            target="N",
            metadata={
            }
        )