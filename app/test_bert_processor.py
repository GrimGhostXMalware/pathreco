import pandas as pd
from bert_processor import BERTProcessor
from data_loader import ONETDataLoader
import time
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

def print_progress(message):
    """Print progress message with timestamp"""
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def test_skill_extraction():
    """Test skill extraction from sample texts"""
    print("\n=== Testing Skill Extraction ===")
    print_progress("Initializing BERT processor...")
    processor = BERTProcessor()
    
    # Test cases
    test_texts = [
        "Experienced in Python programming and machine learning",
        "Proficient in AWS cloud services and DevOps practices",
        "Strong background in data analysis and statistical modeling"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest Case {i}:")
        print(f"Input text: {text}")
        print_progress("Extracting skills...")
        skills = processor.extract_skills(text)
        print("Extracted skills with confidence scores:")
        for skill in skills[:5]:  # Show top 5 skills
            print(f"- {skill['skill']}: {skill['confidence']:.2f}")

def test_sentiment_analysis():
    """Test sentiment analysis on career descriptions"""
    print("\n=== Testing Sentiment Analysis ===")
    print_progress("Loading O*NET data...")
    processor = BERTProcessor()
    data_loader = ONETDataLoader()
    
    # Get sample career descriptions
    occupations = data_loader.get_all_occupations()[:5]  # Test with first 5 occupations
    
    for i, occupation in enumerate(occupations, 1):
        print(f"\nTest Case {i}:")
        print(f"Career: {occupation['Title']}")
        print_progress("Analyzing sentiment...")
        sentiment = processor.analyze_sentiment(occupation['Description'])
        print(f"Sentiment: {sentiment['sentiment']}")
        print(f"Confidence: {sentiment['confidence']:.2f}")

def test_career_matching():
    """Test career matching accuracy"""
    print("\n=== Testing Career Matching ===")
    print_progress("Initializing career matching test...")
    processor = BERTProcessor()
    data_loader = ONETDataLoader()
    
    # Sample user profiles
    test_profiles = [
        {
            "skills": ["Python", "Machine Learning", "Data Analysis"],
            "interests": ["AI", "Research", "Problem Solving"],
            "education": "Bachelor's in Computer Science"
        },
        {
            "skills": ["AWS", "DevOps", "Infrastructure"],
            "interests": ["Cloud Computing", "Automation", "Security"],
            "education": "Bachelor's in Information Technology"
        }
    ]
    
    for i, profile in enumerate(test_profiles, 1):
        print(f"\nTest Profile {i}:")
        print(f"Skills: {profile['skills']}")
        print(f"Interests: {profile['interests']}")
        print(f"Education: {profile['education']}")
        
        print_progress("Getting career matches...")
        # Get career descriptions
        career_descriptions = {}
        for occupation in data_loader.get_all_occupations():
            career_descriptions[occupation['Title']] = occupation['Description']
        
        # Get matches
        matches = processor.match_skills_to_careers(
            profile['skills'],
            career_descriptions
        )
        
        # Show top 3 matches
        print("Top 3 career matches:")
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)[:3]
        for career, score in sorted_matches:
            print(f"- {career}: {score:.2f}")

def test_processing_speed():
    """Test the processing speed of the BERT processor"""
    print("\n=== Testing Processing Speed ===")
    print_progress("Initializing speed test...")
    processor = BERTProcessor()
    data_loader = ONETDataLoader()
    
    # Test with different batch sizes
    batch_sizes = [1, 5, 10]
    
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")
        
        # Get sample data
        occupations = data_loader.get_all_occupations()[:batch_size]
        
        # Test skill extraction
        print_progress("Testing skill extraction speed...")
        start_time = time.time()
        for occupation in occupations:
            processor.extract_skills(occupation['Description'])
        skill_time = time.time() - start_time
        print(f"Skill extraction time: {skill_time:.2f} seconds")
        
        # Test sentiment analysis
        print_progress("Testing sentiment analysis speed...")
        start_time = time.time()
        for occupation in occupations:
            processor.analyze_sentiment(occupation['Description'])
        sentiment_time = time.time() - start_time
        print(f"Sentiment analysis time: {sentiment_time:.2f} seconds")

def main():
    print("Starting BERT Processor Tests...")
    print("=" * 50)
    
    try:
        # Run all tests
        test_skill_extraction()
        test_sentiment_analysis()
        test_career_matching()
        test_processing_speed()
        
        print("\nAll tests completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 