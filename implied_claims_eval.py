"""
Implied Claims Evaluation

This eval tests a model's ability to generate implied claims from a given statement
and then uses an LLM judge to compare generated claims against gold-standard implied claims.
"""

from typing import Optional, Generator, List, Dict, Any
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, scorer, CORRECT, INCORRECT, accuracy,match
from inspect_ai.solver import system_message, generate
import json
import re
from inspect_ai.tool import bash, python
from inspect_ai.agent import Agent, agent, react

CMD_TIMEOUT = 180

@agent
def my_agent(attempts = 1) -> Agent:
    prompt = """Does this paper's abstract seem accurate, such that a naive reader taking the abstract a face value would not be misled?

The paper and preregistration are stored in `/app/files`.

First, generate a list of 10 claims which may be implied by the abstract. Then, determine if each one is supported by the paper. If any claim is determined to be misleading, mark the abstract as misleading.

Submit an answer which consists solely of "Y" if the abstract is misleading or or "N" if it is not misleading.
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
        sandbox=("docker", "compose.yaml"),
        scorer=match()
    )


def create_samples(dataset_file: Optional[str] = None) -> Generator[Sample, None, None]:
    """Create samples from dataset file or default examples."""
    
    # Default sample data if no file provided
    default_samples = [
        {
            "paper": "Wyse_CensusMedicaid.pdf",
            "preregistration": "4hyjr.pdf",
            "target": "Y"
        },
        {
            "paper": "Effect of a Randomized Interventional School-Based Vision Program on Academic Performance of Students in Grades 3 to 7_ A Cluster Randomized Clinical Trial _ Optics and Refraction _ JAMA Ophthalmology _ JAMA Network.pdf",
            "preregistration": "OSF _ VTL analysis plan 06 06 16 RES.docx.pdf",
            "target": "Y"
        },
        {
            "paper": "V07_ASSISTments-tech-report_FINAL-ADA-1.pdf",
            "preregistration": "OSF _ Pre-analysis plan revised 7.15.2020.pdf",
            "target": "N"
        },
        {
            "paper": "long-term-effectiveness-multiple-measures-assessment.pdf",
            "preregistration": "6700.1v2.pdf",
            "target": "Y"
        },
        {
            "paper": "program-effects-fss.pdf",
            "preregistration": "(none)",
            "target": "N"
        },
        {
            "paper": "ssrn-4978991",
            "preregistration": "(none)",
            "target": "N"
        },
        {
            "paper": "NEJMoa2401177.pdf",
            "preregistration": "(none)",
            "target": "N"
        },
    ]
    
    template = """Paper: {paper}
Preregistration plan: {preregistration}"""
    
    for sample_data in default_samples:        
        yield Sample(
            input=template.format(paper= sample_data['paper'], preregistration=sample_data['preregistration']),
            target=sample_data["target"],
            metadata={}
        )