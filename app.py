import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page title and configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App title and description
st.title("Student Performance Prediction")
st.markdown("This app predicts whether a student will graduate, drop out, or remain enrolled based on the top 10 most important features.")

# Create sidebar for inputs
st.sidebar.header("Student Information")

# Input fields for the top 10 features
curricular_units_2nd_sem_approved = st.sidebar.slider("Curricular Units 2nd Sem Approved", 0, 20, 5)
curricular_units_2nd_sem_grade = st.sidebar.slider("Curricular Units 2nd Sem Grade", 0.0, 20.0, 12.0)
curricular_units_1st_sem_approved = st.sidebar.slider("Curricular Units 1st Sem Approved", 0, 20, 5)
curricular_units_1st_sem_grade = st.sidebar.slider("Curricular Units 1st Sem Grade", 0.0, 20.0, 12.0)
tuition_fees_up_to_date = st.sidebar.selectbox("Tuition Fees Up to Date", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
scholarship_holder = st.sidebar.selectbox("Scholarship Holder", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
age_at_enrollment = st.sidebar.slider("Age at Enrollment", 17, 70, 20)
debtor = st.sidebar.selectbox("Debtor", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
gender = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
application_mode = st.sidebar.slider("Application Mode", 1, 17, 1)

# Create a prediction button
if st.sidebar.button("Predict"):
    try:
        # Create input features array
        features = [
            curricular_units_2nd_sem_approved,
            curricular_units_2nd_sem_grade,
            curricular_units_1st_sem_approved,
            curricular_units_1st_sem_grade,
            tuition_fees_up_to_date,
            scholarship_holder,
            age_at_enrollment,
            debtor,
            gender,
            application_mode
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        # Display results
        st.header("Prediction Results")
        
        # Display prediction with color coding
        if prediction == "Graduate":
            st.success(f"Predicted Status: {prediction}")
        elif prediction == "Enrolled":
            st.info(f"Predicted Status: {prediction}")
        else:
            st.error(f"Predicted Status: {prediction}")
        
        # Display prediction probabilities as percentages
        st.subheader("Prediction Probabilities")
        dropout_prob = probability[0] * 100
        enrolled_prob = probability[1] * 100
        graduate_prob = probability[2] * 100
        
        st.write(f"Dropout: {dropout_prob:.1f}%")
        st.write(f"Enrolled: {enrolled_prob:.1f}%")
        st.write(f"Graduate: {graduate_prob:.1f}%")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your input values and try again.")
    
else:
    # Show instructions when the app first loads
    st.write("👈 Adjust the parameters on the sidebar and click 'Predict' to see results.")
    
    # Show dataset information
    st.subheader("About the Model")
    st.write("""
    This model uses the top 10 most important features identified through ANOVA analysis:
    
    1. Curricular Units 2nd Sem Approved
    2. Curricular Units 2nd Sem Grade
    3. Curricular Units 1st Sem Approved
    4. Curricular Units 1st Sem Grade
    5. Tuition Fees Up to Date
    6. Scholarship Holder
    7. Age at Enrollment
    8. Debtor
    9. Gender
    10. Application Mode
    
    These features were found to be the most significant predictors of student performance.
    """)
