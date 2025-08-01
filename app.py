import streamlit as st
import pandas as pd
import joblib

# Load model data
model_data = joblib.load('career_recommendation_model.pkl')
preprocessor = model_data['preprocessor']
rf_model = model_data['rf_model']       # Make sure your RF model is saved here
label_encoder = model_data['label_encoder']

# List of skill columns your model expects
all_skills = [
    'AWS', 'Agile', 'Azure', 'Docker', 'HTML/CSS', 'Java', 'JavaScript', 'Kubernetes',
    'Linux', 'Node.js', 'Pandas', 'Power BI', 'Python', 'React', 'SQL', 'Scrum', 'Spark',
    'Tableau', 'TensorFlow'
]

def prepare_input(input_data):
    # Create base dict with all expected columns
    base = {col: 0 if col in all_skills else "None" for col in preprocessor.feature_names_in_}
    
    # Fill categorical fields
    base['Qualification'] = input_data.get('Qualification', 'None')
    base['Language Proficiency'] = input_data.get('Language Proficiency', 'None')
    base['Previous Internships'] = input_data.get('Previous Internships', 'None')
    base['Certifications'] = input_data.get('Certifications', 'None')

    # Set skill columns to 1 if present in input
    skills_input = input_data.get('Skills', '')
    skills_lower = skills_input.lower()
    for skill in all_skills:
        if skill.lower() in skills_lower:
            base[skill] = 1

    return pd.DataFrame([base])

def recommend_careers_tuned(input_data: dict):
    input_df = prepare_input(input_data)
    input_transformed = preprocessor.transform(input_df)
    probs = rf_model.predict_proba(input_transformed)[0]
    top5_idx = probs.argsort()[::-1][:5]
    return [(label_encoder.classes_[i], probs[i]) for i in top5_idx]

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

def prepare_input(data: dict):
    # Start with base dict for dataframe
    df = pd.DataFrame([{
        'Qualification': data['Qualification'],
        'Language Proficiency': data['Language Proficiency'],
        'Previous Internships': data['Previous Internships'],
        'Certifications': data['Certifications']
    }])

    # Add skill columns (1 if skill selected, else 0)
    for skill in all_skills:
        df[skill] = 1 if skill in data['Skills'] else 0

    return df

if st.button("Recommend Careers"):
    user_input = {
        'Qualification': qualification,
        'Language Proficiency': ", ".join(language_proficiency) if language_proficiency else "None",
        'Previous Internships': ", ".join(previous_internships) if previous_internships else "None",
        'Certifications': ", ".join(certifications_selected) if certifications_selected else "None",
        'Skills': selected_skills
    }

    input_df = prepare_input(user_input)
    # Predict probabilities for all roles
    probs = pipeline.predict_proba(input_df)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    
    st.subheader("🔝 Top 5 Recommended Careers:")
    for idx in top5_idx:
        role = label_encoder.classes_[idx]
        prob = probs[idx]
        st.write(f"**{role}** - Probability: {prob:.2f}")
