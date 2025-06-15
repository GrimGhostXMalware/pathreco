import streamlit as st
import pandas as pd
from predictor import CareerPredictor
from utils import parse_resume
import os

def main():
    st.title("AI Career Advisor")
    st.write("Get personalized career recommendations based on your skills and interests.")
    
    # Input fields
    skills = st.text_input("Enter your skills (comma-separated):", 
                          "python, machine learning, data analysis")
    interests = st.text_input("Enter your interests (comma-separated):", 
                            "technology, problem solving, research")
    education = st.text_input("Enter your education background:", 
                            "Bachelor's in Computer Science")
    
    # Resume upload
    resume_file = st.file_uploader("Upload your resume (PDF or DOCX):", 
                                 type=['pdf', 'docx'])
    
    # Main content area
    if st.button("Get Career Recommendations"):
        with st.spinner("Analyzing your profile..."):
            # Process resume if uploaded
            resume_skills = []
            if resume_file is not None:
                resume_skills = parse_resume(resume_file)
            
            # Combine skills from input and resume
            all_skills = list(set(skills.split(',') + resume_skills))
            
            # Get predictions
            predictor = CareerPredictor()
            recommendations = predictor.predict_careers(
                skills=all_skills,
                interests=interests.split(','),
                education=education
            )
            
            # Display recommendations
            st.header("Your Career Recommendations")
            
            # Sort recommendations by score
            sorted_recommendations = dict(
                sorted(recommendations.items(), 
                      key=lambda x: x[1]['score'], 
                      reverse=True)
            )
            
            if not sorted_recommendations:
                st.warning("No career matches found. Try adding more skills or interests.")
            else:
                for career, details in sorted_recommendations.items():
                    with st.expander(f"**{career}** (Match Score: {details['score']:.2f})"):
                        # Score breakdown
                        st.write("**Match Score Breakdown:**")
                        breakdown = details['score_breakdown']
                        st.write(f"- Skills Match: {breakdown['skills']:.2f} (50% weight)")
                        st.write(f"- Interests Match: {breakdown['interests']:.2f} (25% weight)")
                        st.write(f"- Education Match: {breakdown['education']:.2f} (25% weight)")
                        
                        # Career description
                        st.write("**Description:**")
                        st.write(details['description'])
                        
                        # Sentiment analysis
                        st.write("**Career Outlook:**")
                        sentiment = details['sentiment']
                        st.write(f"Overall sentiment: {sentiment['sentiment']}")
                        st.write(f"Confidence: {sentiment['confidence']:.2f}")
                        
                        # Key phrases
                        st.write("**Key Aspects:**")
                        for phrase in details['key_phrases']:
                            st.write(f"- {phrase}")
                        
                        # Required skills
                        st.write("**Key Required Skills:**")
                        for skill in details['required_skills']:
                            st.write(f"- {skill}")
                        
                        # Matching interests
                        st.write("**Matching Interests:**")
                        for interest in details['interests']:
                            st.write(f"- {interest}")
                        
                        # Additional recommendations
                        st.write("**Recommended Skills to Develop:**")
                        recommended_skills = predictor.get_skill_recommendations(career)
                        for skill_info in recommended_skills:
                            st.write(f"- {skill_info['skill']} (Confidence: {skill_info['confidence']:.2f})")

if __name__ == "__main__":
    main() 