import streamlit as st
import pandas as pd

st.title("Career Path Recommender")
user_skills = st.multiselect("Select your skills:", 
                           ["Python", "JavaScript", "SQL", "Excel"])
if st.button("Get Recommendations"):
    recommendations = recommend_jobs(user_skills, job_data)
    st.write(recommendations)