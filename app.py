import streamlit as st
import pandas as pd


st.title("Career Path Recommender")
user_skills = st.multiselect("Select your skills:", 
                           ["Python", "JavaScript", "SQL", "Excel"])
user_experience = st.number_input("Work Experience (in years):")
if st.button("Get Recommendations"):
    recommendations = recommend_jobs(user_skills, user_experience, job_data)
    st.write(recommendations)   