import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import ONETDataLoader
from bert_processor import BERTProcessor

class CareerPredictor:
    def __init__(self):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Initialize O*NET data loader
        self.data_loader = ONETDataLoader()
        
        # Initialize BERT processor
        self.bert_processor = BERTProcessor()
        
        # Load all occupations
        self.occupations = self.data_loader.get_all_occupations()
        
        # Pre-compute occupation embeddings
        self.occupation_embeddings = self._compute_occupation_embeddings()
    
    def _compute_occupation_embeddings(self):
        """
        Compute embeddings for each occupation using BERT.
        """
        occupation_embeddings = {}
        for occupation in self.occupations:
            # Combine title and description
            text = f"{occupation['Title']} {occupation['Description']}"
            embedding = self.bert_processor.sentence_transformer.encode([text])[0]
            occupation_embeddings[occupation['O*NET-SOC Code']] = embedding
        return occupation_embeddings
    
    def predict_careers(self, skills, interests, education):
        """
        Predict career matches based on user input using BERT.
        
        Args:
            skills (list): List of user skills
            interests (list): List of user interests
            education (str): User's education background
            
        Returns:
            dict: Career recommendations with match scores and insights
        """
        # Get BERT insights
        user_text = f"{' '.join(skills)} {' '.join(interests)} {education}"
        insights = self.bert_processor.extract_career_insights(user_text)
        
        # Calculate similarity scores
        scores = {}
        for occupation_code, occupation_embedding in self.occupation_embeddings.items():
            # Get occupation details
            details = self.data_loader.get_occupation_details(occupation_code)
            occupation_text = f"{details['title']} {details['description']}"
            
            # Calculate component-wise similarities
            skills_text = " ".join(skills)
            interests_text = " ".join(interests)
            
            # Calculate similarities for each component
            skills_similarity = self.bert_processor.compute_similarity(skills_text, occupation_text)
            interests_similarity = self.bert_processor.compute_similarity(interests_text, occupation_text)
            education_similarity = self.bert_processor.compute_similarity(education, occupation_text)
            
            # Calculate overall similarity (weighted average)
            # Skills are weighted more heavily (50%), interests and education 25% each
            overall_similarity = (
                0.5 * skills_similarity +
                0.25 * interests_similarity +
                0.25 * education_similarity
            )
            
            scores[occupation_code] = {
                'overall': float(overall_similarity),
                'components': {
                    'skills': float(skills_similarity),
                    'interests': float(interests_similarity),
                    'education': float(education_similarity)
                }
            }
        
        # Sort by overall score and get top matches
        sorted_scores = dict(sorted(scores.items(), 
                                  key=lambda x: x[1]['overall'], 
                                  reverse=True)[:10])
        
        # Get detailed information for top matches
        recommendations = {}
        for occupation_code, score_info in sorted_scores.items():
            details = self.data_loader.get_occupation_details(occupation_code)
            
            # Get sentiment analysis for career description
            sentiment = self.bert_processor.analyze_sentiment(details['description'])
            
            recommendations[details['title']] = {
                'score': score_info['overall'],
                'score_breakdown': score_info['components'],
                'description': details['description'],
                'required_skills': [skill['Element Name'] for skill in details['skills'] 
                                  if skill['Scale ID'] == 'IM' and float(skill['Data Value']) > 3.0],
                'interests': [interest['Element Name'] for interest in details['interests'] 
                            if interest['Scale ID'] == 'OI' and float(interest['Data Value']) > 3.0],
                'sentiment': sentiment,
                'key_phrases': insights.get('key_phrases', [])
            }
        
        return recommendations
    
    def get_skill_recommendations(self, career_title):
        """
        Get recommended skills for a specific career using BERT.
        
        Args:
            career_title (str): Career title
            
        Returns:
            list: Recommended skills with confidence scores
        """
        # Find occupation code for the given title
        occupation_code = None
        for occupation in self.occupations:
            if occupation['Title'] == career_title:
                occupation_code = occupation['O*NET-SOC Code']
                break
        
        if occupation_code:
            details = self.data_loader.get_occupation_details(occupation_code)
            
            # Get skills with importance level > 3.0
            important_skills = [skill['Element Name'] for skill in details['skills'] 
                              if skill['Scale ID'] == 'IM' and float(skill['Data Value']) > 3.0]
            
            # Get BERT embeddings for skills
            skills_with_scores = []
            for skill in important_skills:
                # Get BERT embedding and confidence score
                skill_embedding = self.bert_processor.sentence_transformer.encode([skill])[0]
                confidence = float(np.mean(skill_embedding))
                
                skills_with_scores.append({
                    'skill': skill,
                    'confidence': confidence
                })
            
            # Sort by confidence score
            return sorted(skills_with_scores, key=lambda x: x['confidence'], reverse=True)
        return [] 