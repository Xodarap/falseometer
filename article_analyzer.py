"""
Article Claim Analyzer using LangChain

This module creates a LangChain agent that analyzes articles by:
1. Reading the article from a URL
2. Splitting the article into sentences
3. For each sentence, generating possible claims
4. For each claim, calculating:
   - Probability someone would interpret the author as making that claim
   - Probability that the claim is true
"""

import os
import re
import json
from typing import List
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


@dataclass
class Claim:
    """Represents a claim extracted from a sentence."""
    text: str
    probability_interpreted: float
    probability_true: float


@dataclass
class SentenceAnalysis:
    """Represents the analysis of a single sentence."""
    sentence: str
    claims: List[Claim]


class ArticleAnalyzer:
    """LangChain agent for analyzing articles and extracting claims."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the analyzer with a language model."""
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in a .env file.\n"
                "You can get an API key from: https://platform.openai.com/api-keys\n"
                "Create a .env file with: OPENAI_API_KEY=your-api-key-here"
            )
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.setup_prompts()
    
    def setup_prompts(self):
        """Set up the prompt templates for different analysis tasks."""
        
        self.claim_extraction_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert at identifying claims made in text. 
            Given a sentence, identify all possible claims that someone might interpret the author as making.
            A claim is a statement that can be true or false.
            
            Return your response as a JSON list of strings, where each string is a potential claim.
            Example: ["The economy is improving", "Unemployment rates are falling"]"""),
            HumanMessage(content="Sentence: {sentence}")
        ])
        
        self.interpretation_probability_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert at understanding how people interpret text.
            Given a sentence and a potential claim, estimate the probability (0-1) that someone would 
            interpret the author as making that claim.
            
            Consider:
            - How directly the sentence states the claim
            - Whether the claim requires inference
            - How reasonable the interpretation is
            
            Return only a single number between 0 and 1."""),
            HumanMessage(content="Sentence: {sentence}\nPotential claim: {claim}")
        ])
        
        self.truth_probability_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert at evaluating the truth of claims.
            Given a claim, estimate the probability (0-1) that the claim is true.
            
            Consider:
            - Available evidence
            - Common knowledge
            - Logical consistency
            - Uncertainty and ambiguity
            
            Return only a single number between 0 and 1."""),
            HumanMessage(content="Claim: {claim}")
        ])
    
    def fetch_article(self, url: str) -> str:
        """Fetch and extract text content from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            raise Exception(f"Error fetching article from {url}: {str(e)}")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Use regex to split on sentence endings, but be careful with abbreviations
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)
        
        # Filter out very short sentences and clean whitespace
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def extract_claims(self, sentence: str) -> List[str]:
        """Extract potential claims from a sentence."""
        try:
            prompt = self.claim_extraction_prompt.format_messages(sentence=sentence)
            response = self.llm(prompt)
            
            # Parse JSON response
            claims_text = response.content.strip()
            if claims_text.startswith('```json'):
                claims_text = claims_text[7:-3]
            elif claims_text.startswith('```'):
                claims_text = claims_text[3:-3]
            
            claims = json.loads(claims_text)
            return claims
            
        except Exception as e:
            print(f"Error extracting claims from sentence: {str(e)}")
            return []
    
    def calculate_interpretation_probability(self, sentence: str, claim: str) -> float:
        """Calculate probability someone would interpret the author as making this claim."""
        try:
            prompt = self.interpretation_probability_prompt.format_messages(
                sentence=sentence, claim=claim
            )
            response = self.llm(prompt)
            
            prob_text = response.content.strip()
            return float(prob_text)
            
        except Exception as e:
            print(f"Error calculating interpretation probability: {str(e)}")
            return 0.5  # Default to neutral probability
    
    def calculate_truth_probability(self, claim: str) -> float:
        """Calculate probability that the claim is true."""
        try:
            prompt = self.truth_probability_prompt.format_messages(claim=claim)
            response = self.llm(prompt)
            
            prob_text = response.content.strip()
            return float(prob_text)
            
        except Exception as e:
            print(f"Error calculating truth probability: {str(e)}")
            return 0.5  # Default to neutral probability
    
    def analyze_article(self, url: str) -> List[SentenceAnalysis]:
        """Analyze an entire article from URL."""
        print(f"Fetching article from: {url}")
        article_text = self.fetch_article(url)
        
        print("Splitting article into sentences...")
        sentences = self.split_into_sentences(article_text)
        print(f"Found {len(sentences)} sentences")
        
        results = []
        
        for i, sentence in enumerate(sentences):
            print(f"\nAnalyzing sentence {i+1}/{len(sentences)}")
            print(f"Sentence: {sentence[:100]}...")
            
            # Extract claims from sentence
            claim_texts = self.extract_claims(sentence)
            print(f"Found {len(claim_texts)} potential claims")
            
            claims = []
            for claim_text in claim_texts:
                print(f"  Analyzing claim: {claim_text}")
                
                # Calculate probabilities
                prob_interpreted = self.calculate_interpretation_probability(sentence, claim_text)
                prob_true = self.calculate_truth_probability(claim_text)
                
                claim = Claim(
                    text=claim_text,
                    probability_interpreted=prob_interpreted,
                    probability_true=prob_true
                )
                claims.append(claim)
                
                print(f"    P(interpreted): {prob_interpreted:.3f}")
                print(f"    P(true): {prob_true:.3f}")
            
            analysis = SentenceAnalysis(sentence=sentence, claims=claims)
            results.append(analysis)
        
        return results
    
    def save_results(self, results: List[SentenceAnalysis], filename: str):
        """Save analysis results to a JSON file."""
        output_data = []
        
        for analysis in results:
            sentence_data = {
                "sentence": analysis.sentence,
                "claims": [
                    {
                        "text": claim.text,
                        "probability_interpreted": claim.probability_interpreted,
                        "probability_true": claim.probability_true
                    }
                    for claim in analysis.claims
                ]
            }
            output_data.append(sentence_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")


def main():
    """Main function to run the article analyzer."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python article_analyzer.py <url>")
        sys.exit(1)
    
    url = sys.argv[1]
    
    analyzer = ArticleAnalyzer()
    results = analyzer.analyze_article(url)
    
    # Save results
    output_filename = "article_analysis_{}.json".format(url.split('/')[-1])
    analyzer.save_results(results, output_filename)
    
    # Print summary
    total_sentences = len(results)
    total_claims = sum(len(analysis.claims) for analysis in results)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Total sentences analyzed: {total_sentences}")
    print(f"Total claims extracted: {total_claims}")
    print(f"Average claims per sentence: {total_claims/total_sentences:.2f}")


if __name__ == "__main__":
    main()