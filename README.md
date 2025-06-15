# AI Career Advisor

An intelligent career guidance system that helps users discover their ideal career paths based on their skills, interests, and background.

## Features

- Career path recommendations based on skills and interests
- Resume parsing and analysis (optional)
- Course and elective suggestions
- Interactive web interface
- ML-powered matching using advanced NLP techniques

## Project Structure

```
ai_career_advisor/
├── app/
│   ├── main.py           # Streamlit UI
│   ├── predictor.py      # ML inference logic
│   └── utils.py          # Resume parser, text processing
├── model/
│   ├── model.pkl         # Trained model
│   └── vectorizer.pkl    # TF-IDF or BERT model
├── data/
│   └── careers_dataset.csv  # Career-skills-interest dataset
├── notebooks/
│   └── model_training.ipynb
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Run the application:
```bash
streamlit run app/main.py
```

## Usage

1. Launch the application using the command above
2. Enter your skills, interests, and academic background
3. Optionally upload your resume
4. Get personalized career recommendations and suggestions

## Technologies Used

- Streamlit for the web interface
- spaCy for NLP and text processing
- Sentence Transformers for semantic similarity
- scikit-learn for ML models
- PyPDF2 and python-docx for resume parsing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 