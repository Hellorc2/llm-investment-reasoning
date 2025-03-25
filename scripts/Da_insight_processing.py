import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional
import json
from datetime import datetime
from pathlib import Path


def extract_key_factors(insight_text: str) -> Dict[str, List[str]]:
    """
    Extract key factors from the insight text and categorize them.
    
    Args:
        insight_text: The raw text from the LLM analysis
        
    Returns:
        Dictionary containing categorized factors
    """
    # Initialize categories
    factors = {
        "technical_background": [],
        "market_timing": [],
        "leadership": [],
        "experience": [],
        "other": []
    }
    
    # TODO: Implement more sophisticated text analysis
    # For now, we'll just split the text into sentences
    sentences = [s.strip() for s in insight_text.split('.') if s.strip()]
    
    # Basic categorization based on keywords
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in ["technical", "engineering", "development", "code"]):
            factors["technical_background"].append(sentence)
        elif any(word in sentence_lower for word in ["market", "timing", "trend", "industry"]):
            factors["market_timing"].append(sentence)
        elif any(word in sentence_lower for word in ["leadership", "management", "vision", "direction"]):
            factors["leadership"].append(sentence)
        elif any(word in sentence_lower for word in ["experience", "background", "previous", "worked"]):
            factors["experience"].append(sentence)
        else:
            factors["other"].append(sentence)
    
    return factors


def save_processed_insights(timestamp: str, raw_insight: str, processed_factors: Dict[str, List[str]]) -> None:
    """
    Save both raw and processed insights to JSON files.
    
    Args:
        timestamp: The timestamp for the session
        raw_insight: The original insight text
        processed_factors: The categorized factors
    """
    # Create records directory if it doesn't exist
    records_dir = Path('records')
    records_dir.mkdir(exist_ok=True)
    
    # Save raw insight
    raw_file = records_dir / f'raw_insight_{timestamp}.json'
    with open(raw_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "raw_insight": raw_insight
        }, f, indent=2)
    
    # Save processed factors
    processed_file = records_dir / f'processed_insight_{timestamp}.json'
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "factors": processed_factors
        }, f, indent=2)


def process_insights(insight_text: str) -> Dict[str, List[str]]:
    """
    Main function to process insights from the LLM analysis.
    
    Args:
        insight_text: The raw text from the LLM analysis
        
    Returns:
        Dictionary containing categorized factors
    """
    # Extract and categorize factors
    factors = extract_key_factors(insight_text)
    
    # Create timestamp for this processing
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save both raw and processed insights
    save_processed_insights(timestamp, insight_text, factors)
    
    return factors


if __name__ == "__main__":
    # Example usage
    sample_insight = """
    The founder had strong technical background with 10 years of software engineering experience.
    They identified a market opportunity in the growing AI industry.
    Their leadership style was collaborative and they had previous startup experience.
    The timing was perfect as the market was ready for their solution.
    """
    
    processed_factors = process_insights(sample_insight)
    print("Processed Factors:")
    print(json.dumps(processed_factors, indent=2))
    

