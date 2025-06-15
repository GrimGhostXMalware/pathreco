import spacy
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import os

class BERTProcessor:
    def __init__(self, batch_size: int = 100):
        """Initialize the BERT processor with necessary models"""
        print("Initializing BERT processor...")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set batch size
        self.batch_size = batch_size
        
        # Force CPU usage
        self.device = torch.device('cpu')
        self.logger.info("Using CPU for all operations")
        
        # Load spaCy model for basic NLP tasks
        self.logger.info("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize sentiment analyzer
        self.logger.info("Loading sentiment analyzer...")
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,  # Force CPU
                batch_size=self.batch_size
            )
        except Exception as e:
            self.logger.error(f"Error loading sentiment analyzer: {str(e)}")
            raise
        
        # Initialize sentence transformer for semantic similarity
        self.logger.info("Loading sentence transformer...")
        try:
            self.sentence_transformer = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device='cpu'  # Force CPU
            )
        except Exception as e:
            self.logger.error(f"Error loading sentence transformer: {str(e)}")
            raise
        
        # Initialize BERT model for additional processing
        self.logger.info("Loading BERT model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Error loading BERT model: {str(e)}")
            raise
        
        self.logger.info("BERT processor initialization complete!")

    def process_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Process a batch of texts for skill extraction"""
        try:
            # Tokenize batch
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Process batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                confidences = torch.mean(embeddings, dim=1).numpy()
            
            return [{'skill': text.lower(), 'confidence': float(conf)} 
                    for text, conf in zip(texts, confidences)]
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            # Return empty results for failed batch
            return [{'skill': text.lower(), 'confidence': 0.0} for text in texts]

    def extract_skills(self, text: str) -> List[Dict[str, float]]:
        """Extract skills from text using spaCy and BERT"""
        doc = self.nlp(text)
        
        # Extract noun phrases and named entities
        potential_skills = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                potential_skills.append(chunk.text)
        
        for ent in doc.ents:
            if ent.label_ in ['SKILL', 'ORG', 'PRODUCT']:
                potential_skills.append(ent.text)
        
        # Process in batches
        skills = []
        for i in range(0, len(potential_skills), self.batch_size):
            batch = potential_skills[i:i + self.batch_size]
            batch_results = self.process_batch(batch)
            skills.extend(batch_results)
        
        return sorted(skills, key=lambda x: x['confidence'], reverse=True)

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                'sentiment': result['label'],
                'confidence': float(result['score'])
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.0}

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using sentence transformer"""
        try:
            # Get embeddings using sentence transformer
            embeddings = self.sentence_transformer.encode([text1, text2], batch_size=self.batch_size)
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                embeddings[0].reshape(1, -1),
                embeddings[1].reshape(1, -1)
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error in similarity computation: {str(e)}")
            return 0.0

    def extract_career_insights(self, text: str) -> Dict[str, List[str]]:
        """Extract career-related insights from text"""
        doc = self.nlp(text)
        
        insights = {
            'skills': [],
            'interests': [],
            'experience': [],
            'key_phrases': []
        }
        
        # Extract potential insights
        potential_insights = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:
                potential_insights.append(chunk.text)
                insights['key_phrases'].append(chunk.text)
        
        # Process in batches
        for i in range(0, len(potential_insights), self.batch_size):
            batch = potential_insights[i:i + self.batch_size]
            batch_results = self.process_batch(batch)
            
            for text, result in zip(batch, batch_results):
                if any(keyword in text.lower() for keyword in ['skill', 'ability', 'proficiency']):
                    insights['skills'].append(text)
                elif any(keyword in text.lower() for keyword in ['interest', 'passion', 'enjoy']):
                    insights['interests'].append(text)
        
        # Extract experience
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in ['experience', 'worked', 'years']):
                insights['experience'].append(sent.text)
                insights['key_phrases'].append(sent.text)
        
        return insights

    def match_skills_to_careers(self, skills: List[str], careers: List[Dict[str, str]], threshold: float = 0.3) -> List[Dict[str, float]]:
        """Match skills to careers using semantic similarity"""
        try:
            # Convert skills list to text
            skills_text = " ".join(skills)
            
            # Get skill embedding
            skill_embedding = self.sentence_transformer.encode([skills_text], batch_size=1)[0]
            
            # Process careers in batches
            matches = []
            for career in tqdm(careers, desc="Processing careers"):
                try:
                    # Combine title and description for better matching
                    career_text = f"{career['title']} {career['description']}"
                    
                    # Get career description embedding
                    career_embedding = self.sentence_transformer.encode([career_text], batch_size=1)[0]
                    
                    # Calculate similarity
                    similarity = cosine_similarity(
                        [skill_embedding],
                        [career_embedding]
                    )[0][0]
                    
                    # Lower threshold for more matches
                    if similarity >= threshold:
                        matches.append({
                            'career': career['title'],
                            'similarity': float(similarity)
                        })
                except Exception as e:
                    self.logger.error(f"Error processing career {career.get('title', 'unknown')}: {str(e)}")
                    continue
            
            # Sort by similarity score
            return sorted(matches, key=lambda x: x['similarity'], reverse=True)
        except Exception as e:
            self.logger.error(f"Error matching skills to careers: {str(e)}")
            return [] 