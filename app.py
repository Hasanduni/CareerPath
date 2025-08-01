import streamlit as st
import pandas as pd
import joblib

import joblib

model_data = joblib.load('career_recommendation_model.pkl')

# Debug: check what is inside model_data
st.write("Type of loaded object:", type(model_data))
if isinstance(model_data, dict):
    st.write("Keys in the loaded dict:", list(model_data.keys()))
else:
    st.write("Loaded object is not a dictionary. It might be a pipeline or model directly.")
preprocessor = model_data['preprocessor']
rf_models_tuned = model_data['models']

def recommend_careers_tuned(input_data: dict):
    input_df = pd.DataFrame([input_data])
    input_processed = preprocessor.transform(input_df)
    
    recommendations = {}
    for role, model in rf_models_tuned.items():
        prob = model.predict_proba(input_processed)[0][1]
        recommendations[role] = prob
    
    ranked_roles = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return ranked_roles

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
    "UI/UX Intern", "Cloud Intern", "Cybersecurity Intern", "BI Intern", "ML Intern","None"
]

certifications = ["AWS Certified", "Azure Certified", "Scrum Master", "None"]

skills = [
    "Python", "Java", "SQL", "JavaScript", "TensorFlow", "Pandas", "Docker", 
    "Kubernetes", "HTML/CSS", "Power BI", "Spark", "AWS", "Azure", 
    "Linux", "Tableau", "React", "Node.js"
]

# --- Blue theme CSS ---
st.markdown("""
    <style>
    /* Primary text and button color */
    .css-1d391kg.edgvbvh3 {color: #0a66c2;} /* Headers */
    .css-1d391kg.edgvbvh3 strong {color: #0a66c2;}
    .stButton>button {background-color: #0a66c2; color: white; border-radius: 8px;}
    .stSelectbox > div, .stMultiSelect > div, .stTextInput > div, .stTextArea > div {
        color: #0a66c2;
    }
    .css-1r6slb0.e1tzin5v1 {
        color: #0a66c2;
    }
    .st-bd {
        color: #0a66c2;
    }
    .stMarkdown p, .stMarkdown span {
        color: #0a66c2;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("🎯 Career Recommendation System")

qualification = st.selectbox("Qualification", qualifications)
language_proficiency = st.multiselect("Language Proficiency (Select one or more)", languages)
previous_internships = st.multiselect("Previous Internships (Select one or more)", internships)
certifications_selected = st.multiselect("Certifications (Select one or more)", certifications)
selected_skills = st.multiselect("Select Your Skills", skills)

if st.button("Recommend Careers"):
    # Prepare input data
    input_data = {
        'Qualification': qualification,
        'Language Proficiency': ", ".join(language_proficiency) if language_proficiency else "None",
        'Previous Internships': ", ".join(previous_internships) if previous_internships else "None",
        'Certifications': ", ".join(certifications_selected) if certifications_selected else "None",
        'Skills': ", ".join(selected_skills) if selected_skills else "None"
    }
    
    results = recommend_careers_tuned(input_data)
    
    st.subheader("🔝 Top 5 Recommended Careers:")
    for role, score in results[:5]:
        st.write(f"**{role}** - Probability: {score:.2f}")

