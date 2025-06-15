import pandas as pd
import numpy as np
from bert_processor import BERTProcessor
from data_loader import ONETDataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict, Tuple, Any
import json
from tqdm import tqdm
import time

class ModelEvaluator:
    def __init__(self, batch_size: int = 100):
        self.processor = BERTProcessor(batch_size=batch_size)
        self.data_loader = ONETDataLoader()
        self.batch_size = batch_size
    
    def evaluate_skill_extraction(self) -> Dict[str, float]:
        """Evaluate skill extraction accuracy"""
        print("\nEvaluating Skill Extraction...")
        results = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        try:
            # Test data
            test_texts = [
                "I have experience with Python, machine learning, and data analysis.",
                "My skills include Java programming, web development, and database management.",
                "I am proficient in C++, software engineering, and system design."
            ]
            true_skills = [
                ["python", "machine learning", "data analysis"],
                ["java", "web development", "database management"],
                ["c++", "software engineering", "system design"]
            ]
            
            # Process texts
            all_predicted_skills = []
            all_true_skills = []
            
            for i, (text, expected_skills) in enumerate(zip(test_texts, true_skills), 1):
                print(f"\nTest Case {i}:")
                print(f"Input text: {text}")
                print(f"Expected skills: {expected_skills}")
                
                # Extract skills
                extracted_skills = self.processor.extract_skills(text)
                # Convert to lowercase for case-insensitive matching
                predicted_skills = [skill['skill'].lower() for skill in extracted_skills]
                
                print(f"Extracted skills: {predicted_skills}")
                
                all_predicted_skills.extend(predicted_skills)
                all_true_skills.extend(expected_skills)
            
            # Calculate metrics
            correct = sum(1 for p, t in zip(all_predicted_skills, all_true_skills) if p == t)
            total = len(all_true_skills)
            results['accuracy'] = correct / total if total > 0 else 0.0
            
            # Calculate precision, recall, and F1
            true_positives = sum(1 for p, t in zip(all_predicted_skills, all_true_skills) if p == t)
            false_positives = sum(1 for p in all_predicted_skills if p not in all_true_skills)
            false_negatives = sum(1 for t in all_true_skills if t not in all_predicted_skills)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results['precision'] = precision
            results['recall'] = recall
            results['f1_score'] = f1
            
            print("\nDetailed Results:")
            print(f"True Positives: {true_positives}")
            print(f"False Positives: {false_positives}")
            print(f"False Negatives: {false_negatives}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            
            return results
            
        except Exception as e:
            print(f"Error in skill extraction evaluation: {str(e)}")
            return results
    
    def evaluate_career_matching(self) -> Dict[str, Any]:
        """Evaluate career matching accuracy"""
        print("\nEvaluating Career Matching...")
        results = {
            'accuracy': 0.0,
            'top_matches': [],
            'processing_time': 0.0,
            'total_careers_evaluated': 0
        }
        
        try:
            # Get sample occupations
            selected_occupations = self.data_loader.get_all_occupations()[:100]  # Use first 100 for testing
            results['total_careers_evaluated'] = len(selected_occupations)
            
            # Test data with different skill sets
            test_profiles = [
                {
                    'skills': ["python", "machine learning", "data analysis"],
                    'expected_careers': ["Data Scientists", "Machine Learning Engineers", "Data Analysts"]
                },
                {
                    'skills': ["java", "spring", "database"],
                    'expected_careers': ["Software Developers", "Database Administrators", "Computer Systems Analysts"]
                },
                {
                    'skills': ["aws", "devops", "kubernetes"],
                    'expected_careers': ["DevOps Engineers", "Cloud Architects", "Systems Administrators"]
                }
            ]
            
            # Measure processing time
            start_time = time.time()
            
            all_accuracies = []
            for profile in test_profiles:
                print(f"\nProcessing profile with skills: {', '.join(profile['skills'])}")
                
                # Convert occupations to the format expected by match_skills_to_careers
                careers = [{'title': occ['Title'], 'description': occ['Description']} 
                          for occ in selected_occupations]
                
                # Get matches
                matches = self.processor.match_skills_to_careers(profile['skills'], careers)
                
                # Get top matches
                top_matches = [match['career'] for match in matches[:3]]
                
                # Calculate accuracy for this profile
                correct_matches = sum(1 for career in profile['expected_careers'] 
                                   if any(career.lower() in match.lower() for match in top_matches))
                profile_accuracy = correct_matches / len(profile['expected_careers'])
                all_accuracies.append(profile_accuracy)
                
                # Store top matches for this profile
                results['top_matches'].append({
                    'profile': profile['skills'],
                    'matches': top_matches,
                    'accuracy': profile_accuracy
                })
                
                print(f"Top matches: {', '.join(top_matches)}")
                print(f"Profile accuracy: {profile_accuracy:.2f}")
            
            # Calculate overall accuracy
            results['accuracy'] = sum(all_accuracies) / len(all_accuracies)
            
            # Calculate processing time
            results['processing_time'] = time.time() - start_time
            
            return results
            
        except Exception as e:
            print(f"Error in career matching evaluation: {str(e)}")
            return results
    
    def evaluate_sentiment_analysis(self) -> Dict[str, float]:
        """Evaluate sentiment analysis accuracy"""
        print("\nEvaluating Sentiment Analysis...")
        results = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        try:
            # Test data
            test_texts = [
                "I love working with data and machine learning!",
                "This job is terrible and stressful.",
                "The work environment is okay, nothing special."
            ]
            true_sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
            
            # Process texts
            predicted_sentiments = []
            for text in tqdm(test_texts, desc="Processing sentiments"):
                sentiment = self.processor.analyze_sentiment(text)
                predicted_sentiments.append(sentiment['sentiment'])
            
            # Calculate metrics
            correct = sum(1 for p, t in zip(predicted_sentiments, true_sentiments) if p == t)
            results['accuracy'] = correct / len(test_texts)
            
            # Calculate precision, recall, and F1 for each sentiment
            for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                true_positives = sum(1 for p, t in zip(predicted_sentiments, true_sentiments) 
                                   if p == sentiment and t == sentiment)
                false_positives = sum(1 for p, t in zip(predicted_sentiments, true_sentiments) 
                                    if p == sentiment and t != sentiment)
                false_negatives = sum(1 for p, t in zip(predicted_sentiments, true_sentiments) 
                                    if p != sentiment and t == sentiment)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results[f'{sentiment.lower()}_precision'] = precision
                results[f'{sentiment.lower()}_recall'] = recall
                results[f'{sentiment.lower()}_f1'] = f1
            
            # Calculate macro averages
            results['precision'] = sum(results[f'{s.lower()}_precision'] for s in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']) / 3
            results['recall'] = sum(results[f'{s.lower()}_recall'] for s in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']) / 3
            results['f1_score'] = sum(results[f'{s.lower()}_f1'] for s in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']) / 3
            
            return results
            
        except Exception as e:
            print(f"Error in sentiment analysis evaluation: {str(e)}")
            return results
    
    def run_all_evaluations(self) -> Dict[str, Any]:
        """Run all model evaluations and return combined results"""
        print("\nStarting Model Evaluation...")
        
        # Run individual evaluations
        skill_results = self.evaluate_skill_extraction()
        career_results = self.evaluate_career_matching()
        sentiment_results = self.evaluate_sentiment_analysis()
        
        # Combine results
        combined_results = {
            'skill_extraction': {
                'accuracy': skill_results['accuracy'],
                'precision': skill_results['precision'],
                'recall': skill_results['recall'],
                'f1_score': skill_results['f1_score']
            },
            'career_matching': {
                'accuracy': career_results['accuracy'],
                'top_matches': career_results['top_matches'],
                'processing_time': career_results['processing_time'],
                'total_careers_evaluated': career_results['total_careers_evaluated']
            },
            'sentiment_analysis': {
                'accuracy': sentiment_results['accuracy'],
                'precision': sentiment_results['precision'],
                'recall': sentiment_results['recall'],
                'f1_score': sentiment_results['f1_score']
            }
        }
        
        # Print summary
        print("\nEvaluation Summary:")
        print("------------------")
        print(f"Skill Extraction - Accuracy: {skill_results['accuracy']:.2f}")
        print(f"Career Matching - Accuracy: {career_results['accuracy']:.2f}")
        print(f"Sentiment Analysis - Accuracy: {sentiment_results['accuracy']:.2f}")
        
        return combined_results

if __name__ == "__main__":
    evaluator = ModelEvaluator(batch_size=100)
    results = evaluator.run_all_evaluations() 