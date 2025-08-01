import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved pipeline and label encoder
model_data = joblib.load('career_recommendation_model.pkl')
pipeline = model_data['pipeline']          # Full pipeline: preprocessing + classifier
label_encoder = model_data['label_encoder']

# List of skill columns expected by the model (binary 0/1)
all_skills = [
    'AWS', 'Agile', 'Azure', 'Docker', 'HTML/CSS', 'Java', 'JavaScript', 'Kubernetes',
    'Linux', 'Node.js', 'Pandas', 'Power BI', 'Python', 'React', 'SQL', 'Scrum', 'Spark',
    'Tableau', 'TensorFlow'
]

# Options
qualifications = [
    "Arts - Information Technology - University of Sri Jayewardenepura",
    "Computer Science - University of Colombo School of Computing (UCSC)",
    "Computer Science - University of Jaffna",
    "Computer Science - University of Ruhuna",
    "Computer Science - Trincomalee Campus, Eastern University, Sri Lanka",
    "Physical Science - ICT - University of Kelaniya",
    "Physical Science - ICT - University of Sri Jayewardenepura",
    "Artificial Intelligence - University of Moratuwa",
    "Electronics and Computer Science - University of Kelaniya",
    "Information Systems - University of Colombo, School of Computing (UCSC)",
    "Information Systems - University of Sri Jayewardenepura",
    "Information Systems - Sabaragamuwa University of Sri Lanka",
    "Data Science - Sabaragamuwa University of Sri Lanaka",
    "Information Technology (IT) - University of Moratuwa",
    "Management and Information Technology (MIT) - University of Kelaniya",
    "Computer Science & Technology - Uva Wellassa University of Sri Lanka",
    "Information Communication Technology - University of Sri Jayewardenepura",
    "Information Communication Technology - University of Kelaniya",
    "Information Communication Technology - University of Vavuniya, Sri Lanka",
    "Information Communication Technology - University of Ruhuna",
    "Information Communication Technology - South Eastern University of Sri Lanka",
    "Information Communication Technology - Rajarata University of Sri Lanka",
    "Information Communication Technology - University of Colombo",
    "Information Communication Technology - Uva Wellassa University of Sri Lanka",
    "Information Communication Technology - Eastern University, Sri Lanka"
]


languages = ['English', 'Sinhala', 'Tamil']

internships = [
    "Software Intern", "Data Analyst Intern", "QA Intern", "Network Intern", 
    "UI/UX Intern", "Cloud Intern", "Cybersecurity Intern", "BI Intern", "ML Intern", "None"
]

certifications = ["AWS Certified", "Azure Certified", "Scrum Master", "None"]

skills = [
    "Python", "Java", "SQL", "JavaScript", "TensorFlow", "Pandas", "Docker", 
    "Kubernetes", "HTML/CSS", "Power BI", "Spark", "AWS", "Azure", 
    "Linux", "Tableau", "React", "Node.js"
]

# Streamlit app title
st.title("🎯 Career Recommendation System")

# User inputs
qualification = st.selectbox("Qualification", qualifications)
language_proficiency = st.multiselect("Language Proficiency (Select one or more)", languages)
previous_internships = st.multiselect("Previous Internships (Select one or more)", internships)
certifications_selected = st.multiselect("Certifications (Select one or more)", certifications)
selected_skills = st.multiselect("Select Your Skills", skills)

def prepare_input(input_data: dict) -> pd.DataFrame:
    # Create DataFrame with categorical columns
    df = pd.DataFrame([{
        'Qualification': input_data['Qualification'],
        'Language Proficiency': input_data['Language Proficiency'],
        'Previous Internships': input_data['Previous Internships'],
        'Certifications': input_data['Certifications']
    }])

    # Add skill columns as binary indicators (1 if selected, else 0)
    for skill in all_skills:
        df[skill] = 1 if skill in input_data['Skills'] else 0

    return df

def recommend_careers(input_data: dict):
    df = prepare_input(input_data)
    probs = pipeline.predict_proba(df)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    return [(label_encoder.classes_[i], probs[i]) for i in top5_idx]

if st.button("Recommend Careers"):
    user_input = {
        'Qualification': qualification,
        'Language Proficiency': ", ".join(language_proficiency) if language_proficiency else "None",
        'Previous Internships': ", ".join(previous_internships) if previous_internships else "None",
        'Certifications': ", ".join(certifications_selected) if certifications_selected else "None",
        'Skills': selected_skills
    }

    recommendations = recommend_careers(user_input)

    st.subheader("🔝 Top 5 Recommended Careers:")
    for role, prob in recommendations:
        st.write(f"**{role}** - Probability: {prob:.2f}")
