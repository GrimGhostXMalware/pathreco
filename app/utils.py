import spacy
import PyPDF2
from docx import Document
import re
from typing import List, Union

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def parse_resume(file) -> List[str]:
    """
    Parse resume file (PDF or DOCX) and extract skills.
    
    Args:
        file: Uploaded file object (PDF or DOCX)
        
    Returns:
        List of extracted skills
    """
    # Read file content based on type
    if file.name.endswith('.pdf'):
        text = _read_pdf(file)
    elif file.name.endswith('.docx'):
        text = _read_docx(file)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or DOCX.")
    
    # Extract skills from text
    return extract_skills(text)

def _read_pdf(file) -> str:
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def _read_docx(file) -> str:
    """Extract text from DOCX file."""
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_skills(text: str) -> List[str]:
    """
    Extract skills from text using NLP.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted skills
    """
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract noun phrases and named entities
    skills = set()
    
    # Common technical skills patterns
    skill_patterns = [
        r'\b(?:Python|Java|JavaScript|C\+\+|Ruby|PHP|Swift|Kotlin)\b',
        r'\b(?:Machine Learning|AI|Deep Learning|NLP|Computer Vision)\b',
        r'\b(?:AWS|Azure|GCP|Cloud Computing)\b',
        r'\b(?:SQL|NoSQL|Database|MongoDB|PostgreSQL)\b',
        r'\b(?:React|Angular|Vue|Node\.js|Django|Flask)\b',
        r'\b(?:DevOps|CI/CD|Docker|Kubernetes)\b',
        r'\b(?:Git|GitHub|GitLab|Bitbucket)\b',
        r'\b(?:Agile|Scrum|Kanban)\b',
        r'\b(?:Data Analysis|Data Science|Big Data)\b',
        r'\b(?:UI/UX|User Interface|User Experience)\b'
    ]
    
    # Extract skills using patterns
    for pattern in skill_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            skills.add(match.group())
    
    # Extract noun phrases that might be skills
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:  # Limit to phrases of 3 words or less
            skills.add(chunk.text)
    
    # Extract named entities that might be skills
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'ORG', 'WORK_OF_ART']:
            skills.add(ent.text)
    
    return list(skills) 