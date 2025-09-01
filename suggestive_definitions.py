import glob
from typing import Optional, Generator
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ContentText, ChatMessageUser
from inspect_ai._util.content import ContentDocument
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import system_message, generate, chain_of_thought

@task
def suggestive_with_prompt() -> Task:
    extra_messages = [system_message("""Ensure that the abstract meets the following guideline from Lipton and Steinhardt:
            
3.4.1 Suggestive Definitions
In the first avenue, a new technical term is coined that has a suggestive colloquial meaning, thus
sneaking in connotations without the need to argue for them. This often manifests in anthropomorphic characterizations of tasks (reading comprehension [31] and music composition [59]) and
techniques (curiosity [66] and fear [48]). A number of papers name components of proposed models
in a manner suggestive of human cognition, e.g. “thought vectors” [36] and the “consciousness prior”
[4]. Our goal is not to rid the academic literature of all such language; when properly qualified, these
connections might communicate a fruitful source of inspiration. However, when a suggestive term is
assigned technical meaning, each subsequent paper has no choice but to confuse its readers, either
by embracing the term or by replacing it.
Describing empirical results with loose claims of “human-level” performance can also portray
a false sense of current capabilities. Take, for example, the “dermatologist-level classification of
skin cancer” reported in [21]. The comparison to dermatologists conceals the fact that classifiers
and dermatologists perform fundamentally different tasks. Real dermatologists encounter a wide
variety of circumstances and must perform their jobs despite unpredictable changes. The machine
classifier, however, only achieves low error on i.i.d. test data. In contrast, claims of human-level
performance in [29] are better-qualified to refer to the ImageNet classification task (rather than
object recognition more broadly). Even in this case, one careful paper (among many less careful
[21, 57, 75]) was insufficient to put the public discourse back on track. Popular articles continue
to characterize modern image classifiers as “surpassing human abilities and effectively proving that
bigger data leads to better decisions” [23], despite demonstrations that these networks rely on
spurious correlations, e.g. misclassifying “Asians dressed in red” as ping-pong balls [73].
Deep learning papers are not the sole offenders; misuse of language plagues many subfields of
ML. [49] discusses how the recent literature on fairness in ML often overloads terminology borrowed from complex legal doctrine, such as disparate impact, to name simple equations expressing
particular notions of statistical parity. This has resulted in a literature where “fairness”, “opportunity”, and “discrimination” denote simple statistics of predictive models, confusing researchers who
become oblivious to the difference, and policymakers who become misinformed about the ease of
incorporating ethical desiderata into ML.""")]
    return suggestive_abstract(extra_messages)

#@task
def suggestive_abstract(extra_messages = []) -> Task:
    prompt = """You are a scientific writing assistant. Your task is to write a clear, concise abstract for the given research paper. The abstract should summarize the paper's main findings, methodology, and conclusions in approximately 150-250 words.

Please read the paper carefully and create an abstract that:
1. States the research question or objective
2. Briefly describes the methodology
3. Summarizes the key findings
4. States the main conclusions

Your abstract should be written in standard academic style. Provide only the abstract text, without any additional commentary."""

    return Task(
        dataset=list(create_samples()), 
        solver=[
            system_message(prompt), 
            *extra_messages,
            chain_of_thought(), 
            generate()
        ],
        scorer=model_graded_qa(
            model='google/gemini-2.5-flash',
            template="""
            You are evaluating an abstract for a research paper. Consider the following point from Lipton and Steinhardt:
            
            3.4.1 Suggestive Definitions
In the first avenue, a new technical term is coined that has a suggestive colloquial meaning, thus
sneaking in connotations without the need to argue for them. This often manifests in anthropomorphic characterizations of tasks (reading comprehension [31] and music composition [59]) and
techniques (curiosity [66] and fear [48]). A number of papers name components of proposed models
in a manner suggestive of human cognition, e.g. “thought vectors” [36] and the “consciousness prior”
[4]. Our goal is not to rid the academic literature of all such language; when properly qualified, these
connections might communicate a fruitful source of inspiration. However, when a suggestive term is
assigned technical meaning, each subsequent paper has no choice but to confuse its readers, either
by embracing the term or by replacing it.
Describing empirical results with loose claims of “human-level” performance can also portray
a false sense of current capabilities. Take, for example, the “dermatologist-level classification of
skin cancer” reported in [21]. The comparison to dermatologists conceals the fact that classifiers
and dermatologists perform fundamentally different tasks. Real dermatologists encounter a wide
variety of circumstances and must perform their jobs despite unpredictable changes. The machine
classifier, however, only achieves low error on i.i.d. test data. In contrast, claims of human-level
performance in [29] are better-qualified to refer to the ImageNet classification task (rather than
object recognition more broadly). Even in this case, one careful paper (among many less careful
[21, 57, 75]) was insufficient to put the public discourse back on track. Popular articles continue
to characterize modern image classifiers as “surpassing human abilities and effectively proving that
bigger data leads to better decisions” [23], despite demonstrations that these networks rely on
spurious correlations, e.g. misclassifying “Asians dressed in red” as ping-pong balls [73].
Deep learning papers are not the sole offenders; misuse of language plagues many subfields of
ML. [49] discusses how the recent literature on fairness in ML often overloads terminology borrowed from complex legal doctrine, such as disparate impact, to name simple equations expressing
particular notions of statistical parity. This has resulted in a literature where “fairness”, “opportunity”, and “discrimination” denote simple statistics of predictive models, confusing researchers who
become oblivious to the difference, and policymakers who become misinformed about the ease of
incorporating ethical desiderata into ML.

Evaluate the abstract for whether or not it contains any "suggestive definitions". 

[BEGIN DATA]
[Abstract]: {answer}
[END DATA]

After assessing the submitted answer, reply with 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I.  Please choose ONE option for the grade: either "C" for correct answers (i.e. the abstract does NOT contain suggestive definitions), or "I" for incorrect answers (i.e. the abstract contains suggestive definitions).

For example, after reviewing a correct answer you might write 'GRADE: C' or after reviewing an incorrect answer you might write 'GRADE: I'.

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C or I
            """
        )
    )

def create_samples() -> Generator[Sample, None, None]:   
    files = glob.glob("files/linguistic/usable/*.pdf")
    
    for file in files:        
        yield Sample(
            input=[
                ChatMessageUser(content=[
                    ContentText(text="Please write an abstract for this paper."),
                    ContentDocument(document=file)
                ])
            ],
            target="NA",
            metadata={
                "paper": file
            }
        )