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
    interpretation_explanation: str
    truth_explanation: str
    microlies: float


@dataclass
class SentenceAnalysis:
    """Represents the analysis of a single sentence."""
    sentence: str
    claims: List[Claim]
    sentence_microlies: float


class ArticleAnalyzer:
    """LangChain agent for analyzing articles and extracting claims."""
    
    def __init__(self, model_name: str = "gpt-4.1-mini"):
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
            ("system", """You are an expert at identifying claims made in text. 
            Given a sentence, identify all possible claims that someone might interpret the author as making.
            A claim is a statement that can be true or false.
            
            Return your response as a JSON list of strings, where each string is a potential claim.
            Example: ["The economy is improving", "Unemployment rates are falling"]"""),
            ("human", "Sentence: {sentence}")
        ])
        
        self.interpretation_probability_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at understanding how people interpret text.
            Given a sentence from an article and a potential claim, estimate the probability (0-1) that someone would 
            interpret the author as making that claim.
            
            Consider:
            - How directly the sentence states the claim
            - Whether the claim requires inference
            - How reasonable the interpretation is
            - The context provided by the full article
            
            Format your response as JSON with this structure:
            {{
                "explanation": "Brief explanation of your reasoning",
                "probability": 0.75
            }}
            
            The probability must be a decimal between 0 and 1."""),
            ("human", "Full article text:\n{article_text}\n\nSpecific sentence: {sentence}\nPotential claim: {claim}")
        ])
        
        self.truth_probability_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at evaluating the truth of claims.
            Given a claim from an article, estimate the probability (0-1) that the claim is true.
            
            Consider:
            - Available evidence in the article
            - Common knowledge
            - Logical consistency
            - Uncertainty and ambiguity
            - Context provided by the full article
            
            Format your response as JSON with this structure:
            {{
                "explanation": "Brief explanation of your reasoning",
                "probability": 0.75
            }}
            
            The probability must be a decimal between 0 and 1."""),
            ("human", "Full article text:\n{article_text}\n\nClaim to evaluate: {claim}")
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
            response = self.llm.invoke(prompt)
            
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
    
    def calculate_interpretation_probability(self, sentence: str, claim: str, article_text: str) -> tuple[float, str]:
        """Calculate probability someone would interpret the author as making this claim."""
        try:
            prompt_messages = self.interpretation_probability_prompt.format_messages(
                article_text=article_text, sentence=sentence, claim=claim
            )
            
            response = self.llm.invoke(prompt_messages)
            response_text = response.content.strip()
            
            return self._parse_probability_response(response_text)
            
        except Exception as e:
            print(f"Error calculating interpretation probability: {str(e)}")
            return 0.5, f"Error: {str(e)}"
    
    def calculate_truth_probability(self, claim: str, article_text: str) -> tuple[float, str]:
        """Calculate probability that the claim is true."""
        try:
            prompt_messages = self.truth_probability_prompt.format_messages(
                article_text=article_text, claim=claim
            )
            response = self.llm.invoke(prompt_messages)
            response_text = response.content.strip()
            
            return self._parse_probability_response(response_text)
            
        except Exception as e:
            print(f"Error calculating truth probability: {str(e)}")
            return 0.5, f"Error: {str(e)}"
    
    def _parse_probability_response(self, response_text: str) -> tuple[float, str]:
        """Parse probability and explanation from LLM response."""
        try:
            # Clean up potential markdown formatting
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            
            # Try to find JSON object in the text
            import re
            json_match = re.search(r'\{[^{}]*"explanation"[^{}]*"probability"[^{}]*\}', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                prob = float(result.get("probability", 0.5))
                explanation = result.get("explanation", "No explanation provided")
                return max(0.0, min(1.0, prob)), explanation
            else:
                # Try parsing the whole cleaned text
                result = json.loads(cleaned_text)
                prob = float(result.get("probability", 0.5))
                explanation = result.get("explanation", "No explanation provided")
                return max(0.0, min(1.0, prob)), explanation
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  DEBUG - JSON parse error: {e}")
            print(f"  DEBUG - Response text: '{response_text}'")
            
            # Fallback: try to extract number from text
            import re
            numbers = re.findall(r'0?\.\d+|[01]\.?\d*', response_text)
            if numbers:
                prob = float(numbers[0])
                return max(0.0, min(1.0, prob)), f"Fallback parse: {response_text}"
            else:
                return 0.5, f"Parse failed: {response_text}"
    
    def analyze_article(self, url: str, max_sentences: int = None, max_claims: int = None, skip_sentences: int = 0) -> List[SentenceAnalysis]:
        """Analyze an entire article from URL."""
        print(f"Fetching article from: {url}")
        article_text = self.fetch_article(url)
        
        print("Splitting article into sentences...")
        sentences = self.split_into_sentences(article_text)
        
        # Skip sentences if specified
        if skip_sentences > 0:
            sentences = sentences[skip_sentences:]
            print(f"Skipping first {skip_sentences} sentences")
        
        # Limit sentences if specified
        if max_sentences:
            sentences = sentences[:max_sentences]
            print(f"Limiting to first {max_sentences} sentences")
        
        print(f"Found {len(sentences)} sentences to analyze")
        
        results = []
        
        for i, sentence in enumerate(sentences):
            print(f"\nAnalyzing sentence {i+1}/{len(sentences)}")
            print(f"Sentence: {sentence[:100]}...")
            
            # Extract claims from sentence
            claim_texts = self.extract_claims(sentence)
            
            # Limit claims if specified
            if max_claims:
                claim_texts = claim_texts[:max_claims]
                print(f"Found {len(claim_texts)} potential claims (limited to {max_claims})")
            else:
                print(f"Found {len(claim_texts)} potential claims")
            
            claims = []
            for claim_text in claim_texts:
                print(f"  Analyzing claim: {claim_text}")
                
                # Calculate probabilities with explanations
                prob_interpreted, interp_explanation = self.calculate_interpretation_probability(sentence, claim_text, article_text)
                prob_true, truth_explanation = self.calculate_truth_probability(claim_text, article_text)
                
                # Calculate microlies: (p(claim made) * p(claim false))^3
                prob_false = 1.0 - prob_true
                microlies = (prob_interpreted * prob_false) ** 3 * 1000000
                
                claim = Claim(
                    text=claim_text,
                    probability_interpreted=prob_interpreted,
                    probability_true=prob_true,
                    interpretation_explanation=interp_explanation,
                    truth_explanation=truth_explanation,
                    microlies=microlies
                )
                claims.append(claim)
                
                print(f"    P(interpreted): {prob_interpreted:.3f}")
                print(f"    P(true): {prob_true:.3f}")
                print(f"    Microlies: {microlies:.6f}")
            
            # Calculate sentence-level microlies (sum of all claim microlies)
            sentence_microlies = sum(claim.microlies for claim in claims)
            
            analysis = SentenceAnalysis(
                sentence=sentence, 
                claims=claims,
                sentence_microlies=sentence_microlies
            )
            results.append(analysis)
            
            print(f"  Sentence microlies total: {sentence_microlies:.6f}")
        
        return results
    
    def save_results(self, results: List[SentenceAnalysis], filename: str):
        """Save analysis results to a JSON file."""
        sentences_data = []
        
        for analysis in results:
            sentence_data = {
                "sentence": analysis.sentence,
                "sentence_microlies": analysis.sentence_microlies,
                "claims": [
                    {
                        "text": claim.text,
                        "probability_interpreted": claim.probability_interpreted,
                        "probability_true": claim.probability_true,
                        "interpretation_explanation": claim.interpretation_explanation,
                        "truth_explanation": claim.truth_explanation,
                        "microlies": claim.microlies
                    }
                    for claim in analysis.claims
                ]
            }
            sentences_data.append(sentence_data)
        
        # Calculate article-level microlies (sum of all sentence microlies)
        article_microlies = sum(analysis.sentence_microlies for analysis in results)
        
        output_data = {
            "article_microlies": article_microlies,
            "total_sentences": len(results),
            "total_claims": sum(len(analysis.claims) for analysis in results),
            "sentences": sentences_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
        print(f"Article microlies total: {article_microlies:.6f}")


def main():
    """Main function to run the article analyzer."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Analyze articles for claims")
    parser.add_argument("url", help="URL of the article to analyze")
    parser.add_argument("--sentences", type=int, help="Limit number of sentences to analyze")
    parser.add_argument("--claims", type=int, help="Limit number of claims to analyze per sentence")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N sentences")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("analysis_log", exist_ok=True)
    
    analyzer = ArticleAnalyzer()
    results = analyzer.analyze_article(args.url, max_sentences=args.sentences, max_claims=args.claims, skip_sentences=args.skip)
    
    # Save results in analysis_log folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    article_id = args.url.split('/')[-1]
    output_filename = os.path.join("analysis_log", f"article_analysis_{article_id}_{timestamp}.json")
    analyzer.save_results(results, output_filename)
    
    # Print summary
    total_sentences = len(results)
    total_claims = sum(len(analysis.claims) for analysis in results)
    total_microlies = sum(analysis.sentence_microlies for analysis in results)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Total sentences analyzed: {total_sentences}")
    print(f"Total claims extracted: {total_claims}")
    print(f"Total article microlies: {total_microlies:.6f}")
    if total_sentences > 0:
        print(f"Average claims per sentence: {total_claims/total_sentences:.2f}")
        print(f"Average microlies per sentence: {total_microlies/total_sentences:.6f}")
    if total_claims > 0:
        print(f"Average microlies per claim: {total_microlies/total_claims:.6f}")


if __name__ == "__main__":
    main()