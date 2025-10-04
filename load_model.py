import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the model from file
def load_model():
  try:
      with open('best_model.pkl', 'rb') as f:
          model = pickle.load(f)
      return model
  except Exception as e:
      st.error(f"Error loading model: {e}")
      return None

# Streamlit UI for predictions
def main():
  st.title("Model Prediction App")
  
  model = load_model()
  if not model:
      st.stop()
  
  # Display information about the model
  st.write("## Model Information")
  model_type = type(model).__name__
  st.write(f"Model type: {model_type}")
  
  # Create input fields for each feature
  st.write("## Enter Feature Values")
  
  # Get user inputs
  user_inputs = {
    'age': st.number_input('Enter age', value=0.0),
    'gender': st.number_input('Enter gender', value=0.0),
    'bmi': st.number_input('Enter bmi', value=0.0),
    'blood_pressure': st.number_input('Enter blood_pressure', value=0.0),
    'cholesterol_level': st.number_input('Enter cholesterol_level', value=0.0),
    'glucose_level': st.number_input('Enter glucose_level', value=0.0),
    'physical_activity': st.number_input('Enter physical_activity', value=0.0),
    'smoking_status': st.number_input('Enter smoking_status', value=0.0),
    'alcohol_intake': st.number_input('Enter alcohol_intake', value=0.0),
    'family_history': st.number_input('Enter family_history', value=0.0),
    'biomarker_A': st.number_input('Enter biomarker_A', value=0.0),
    'biomarker_B': st.number_input('Enter biomarker_B', value=0.0),
    'biomarker_C': st.number_input('Enter biomarker_C', value=0.0),
    'biomarker_D': st.number_input('Enter biomarker_D', value=0.0),

  }
  
  # Predict the output
  if st.button("Predict"):
      try:
          # Create a DataFrame with the input values
          input_df = pd.DataFrame([user_inputs])
          
          # Make prediction
          prediction = model.predict(input_df)
          
          # Display the prediction
          st.write("## Prediction Result")
          
          # Check if it's a classification or regression model
          if hasattr(model, 'classes_'):
              # Classification model
              st.write(f"Predicted class: {prediction[0]}")
              
              # If model has predict_proba method, show probabilities
              if hasattr(model, 'predict_proba'):
                  try:
                      proba = model.predict_proba(input_df)
                      st.write("### Class Probabilities")
                      for i, class_name in enumerate(model.classes_):
                          st.write(f"{class_name}: {proba[0][i]:.4f}")
                  except:
                      pass
          else:
              # Regression model
              st.write(f"Predicted value: {prediction[0]:.4f}")
              
      except Exception as e:
          st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
  main()