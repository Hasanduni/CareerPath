import streamlit as st
import pandas as pd
import joblib

# Load saved pipeline and label encoder
model_data = joblib.load('career_recommendation_model.pkl')


preprocessor = model_data['preprocessor']       # pipeline including preprocessing + model
label_encoder = model_data['label_encoder']     # label encoder for Role
st.write("Expected columns in preprocessor:")
try:
    st.write(preprocessor.feature_names_in_)
except Exception as e:
    st.write("Error getting feature_names_in_:", e)

def recommend_careers_tuned(input_data: dict):
    input_df = pd.DataFrame([input_data])
    # Predict probabilities for all classes directly using pipeline
    probs = preprocessor.predict_proba(input_df)[0]
    
    # Get indices of top 5 probabilities
    top5_idx = probs.argsort()[::-1][:5]
    
    # Map indices to role names and probabilities
    top5_roles = [(label_encoder.classes_[i], probs[i]) for i in top5_idx]
    return top5_roles

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

# Blue theme CSS
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
    input_data = {
        'Qualification': qualification,
        'Language Proficiency': ", ".join(language_proficiency) if language_proficiency else "None",
        'Previous Internships': ", ".join(previous_internships) if previous_internships else "None",
        'Certifications': ", ".join(certifications_selected) if certifications_selected else "None",
        'Skills': ", ".join(selected_skills) if selected_skills else "None"
    }
    
    results = recommend_careers_tuned(input_data)
    
    st.subheader("🔝 Top 5 Recommended Careers:")
    for role, score in results:
        st.write(f"**{role}** - Probability: {score:.2f}")


