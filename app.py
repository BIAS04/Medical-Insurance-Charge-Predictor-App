import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Import joblib

# --- CSS for aesthetics ---
def local_css():
    css = """
    <style>
        /* --- Make the logo in the sidebar circular --- */
        [data-testid="stSidebar"] [data-testid="stImage"] img {
            border-radius: 50%;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* --- Style the main "Predict" button --- */
        [data-testid="stButton"] button {
            border-radius: 12px;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        [data-testid="stButton"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }

        /* --- Create "Cards" for the results --- */
        
        /* Card 1: The Prediction Metric */
        [data-testid="stMetric"] {
            /* --- CHANGED: Use theme-aware background --- */
            background-color: var(--secondary-background-color); 
            border: 1px solid var(--gray-200, #E0E0E0);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        [data-testid="stMetric"]:hover {
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
        }
        [data-testid="stMetricValue"] {
            /* Use the green theme color */
            color: #22c55e; 
            font-size: 2.25rem;
            font-weight: 600;
        }

        /* Card 2: The Inputs Table */
        [data-testid="stDataFrame"] {
            /* --- CHANGED: Use theme-aware background --- */
            background-color: var(--secondary-background-color);
            border: 1px solid var(--gray-200, #E0E0E0);
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        
        /* --- Add some subtle spacing to sidebar widgets --- */
        [data-testid="stSidebar"] [data-testid*="stSlider"] {
            padding-bottom: 0.75rem;
        }
        [data-testid="stSidebar"] [data-testid*="stRadio"] {
            padding-bottom: 0.75rem;
        }
        [data-testid="stSidebar"] [data-testid*="stNumberInput"] {
            padding-bottom: 0.75rem;
        }
        [data-testid="stSidebar"] [data-testid*="stSelectbox"] {
            padding-bottom: 0.75rem;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Functions to load model components ---
# (These functions are unchanged)
@st.cache_resource
def load_model():
    """Loads the saved Linear Regression model."""
    try:
        model = joblib.load('linear_regression_model_joblib.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'linear_regression_model_joblib.pkl' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Loads the saved scaler."""
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except FileNotFoundError:
        st.error("Scaler file 'scaler.pkl' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

@st.cache_resource
def load_columns():
    """Loads the saved column list."""
    try:
        columns = joblib.load('columns.pkl')
        return columns
    except FileNotFoundError:
        st.error("Columns file 'columns.pkl' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading columns: {e}")
        return None

# --- Feature Engineering Function ---
# (This function is unchanged)
def preprocess_inputs(age, sex, bmi, children, smoker, region):
    """
    Takes raw user inputs and converts them into the one-hot encoded
    feature vector your model expects.
    """
    
    # Initialize dictionary for features
    features = {
        'age': age,
        'is_female': 1 if sex == 'Female' else 0,
        'bmi': bmi,
        'children': children,
        'is_smoker': 1 if smoker == 'Yes' else 0,
        'charges': 0,  # Placeholder for the target variable, not used in prediction
        'region_northwest': 0,
        'region_southeast': 0,
        'region_southwest': 0,
        'bmi_category_Normal weight': 0,
        'bmi_category_Overweight': 0,
        'bmi_category_Obesity': 0
    }
    
    # --- One-hot encode region ---
    # Note: 'northeast' is the base case (all region flags are 0)
    if region == 'Northwest':
        features['region_northwest'] = 1
    elif region == 'Southeast':
        features['region_southeast'] = 1
    elif region == 'Southwest':
        features['region_southwest'] = 1
        
    # --- One-hot encode BMI category ---
    # Note: 'Underweight' (< 18.5) is the base case
    if 18.5 <= bmi < 25:
        features['bmi_category_Normal weight'] = 1
    elif 25 <= bmi < 30:
        features['bmi_category_Overweight'] = 1
    elif bmi >= 30:
        features['bmi_category_Obesity'] = 1
        
    # Column names for the DataFrame (excluding 'charges')
    # This list *must* contain all keys from the 'features' dict 
    # (except 'charges') that your 'columns.pkl' might expect.
    column_names = [
        'age', 'is_female', 'bmi', 'children', 'is_smoker',
        'region_northwest', 'region_southeast', 'region_southwest',
        'bmi_category_Normal weight', 'bmi_category_Overweight',
        'bmi_category_Obesity'
    ]
    
    # Create a dictionary with only the features
    final_features_dict = {col: features[col] for col in column_names}
    
    # Convert to DataFrame
    features_df = pd.DataFrame(final_features_dict, index=[0])
    
    return features_df

# --- Prediction Function ---
# (This function is unchanged)
def make_prediction(model, scaler, columns, input_df):
    """
    Uses the loaded model, scaler, and column list to make a prediction.
    """
    if model is None or scaler is None or columns is None:
        st.error("Model, scaler, or columns not loaded. Cannot predict.")
        return None
        
    try:
        # 1. Ensure columns are in the correct order (as per columns.pkl)
        # This DataFrame now has all columns the *model* expects.
        input_df_ordered = input_df[columns]
        
        # 2. Identify numerical columns the *scaler* was fit on.
        numerical_cols = ['age', 'bmi', 'children']

        # 3. Create a copy to hold the scaled data
        input_df_scaled = input_df_ordered.copy()

        # 4. Scale *only* the numerical columns
        input_df_scaled[numerical_cols] = scaler.transform(input_df_ordered[numerical_cols])
        
        # 5. Make prediction using the fully prepared DataFrame
        prediction = model.predict(input_df_scaled)
        
        # Return the prediction, the *ordered* (pre-scale) df, and the *scaled* df
        return prediction[0], input_df_ordered, input_df_scaled
        
    except KeyError as e:
        st.error(f"Column mismatch error: {e}. The app's features do not match 'columns.pkl'.")
        return None, input_df, None
    except ValueError as e:
        st.error(f"Scaling error: {e}. Check the 'numerical_cols' list in 'make_prediction'.")
        return None, input_df_ordered, None
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, input_df, None

# --- Streamlit App UI ---

# Set page config
st.set_page_config(
    page_title="🩺 Insurance Charge Predictor", 
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar open
)

# --- Apply Custom CSS ---
local_css()

# --- Sidebar for User Inputs ---
with st.sidebar:
    # Use the green theme color for the logo
    st.image(
        "https://placehold.co/150x150/22c55e/FFFFFF?text=MediPredict&font=inter", 
        width=100
    )
    st.title("👤 Patient Details")
    st.write("Enter patient information below to predict medical insurance charges.")

    # Input: Age
    age = st.slider("Age", 18, 100, 30, help="Patient's age in years.")

    # Input: Sex
    sex = st.radio("Sex", ("Male", "Female"), help="Patient's biological sex.")

    # Input: BMI
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0, 0.1, help="Body Mass Index (kg/m^2).")

    # Input: Children
    children = st.slider("Number of Children", 0, 5, 0, help="Number of dependents.")

    # Input: Smoker
    smoker = st.radio("Smoker?", ("No", "Yes"), help="Does the patient smoke?")

    # Input: Region
    region = st.selectbox(
        "Region",
        ("Southwest", "Southeast", "Northwest", "Northeast"),
        help="Patient's residential region in the US."
    )
    
    st.markdown("---")
    st.info("This app uses a Linear Regression model to predict charges.")


# --- Main Panel ---
st.title("🩺 Medical Insurance Charge Predictor")
st.markdown(
    """
    Welcome! This tool predicts medical insurance charges based on patient demographics and health factors. 
    The prediction is generated by a machine learning model.
    """
)
st.markdown("---")

# Preprocess inputs when button is clicked
if st.button("Predict Charges 💰", use_container_width=True, type="primary"):
    
    # Load model components
    model = load_model()
    scaler = load_scaler()
    model_columns = load_columns()
    
    # 1. Preprocess the raw inputs
    features_df = preprocess_inputs(age, sex, bmi, children, smoker, region)
    
    # 2. Make a prediction
    if model and scaler and model_columns:
        with st.spinner("Running prediction model..."):
            result = make_prediction(model, scaler, model_columns, features_df)
        
        if result and result[0] is not None:
            prediction, features_ordered, features_scaled = result
            
            # --- Display Results in Columns ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Result")
                st.metric(label="Predicted Insurance Charge", value=f"${prediction:,.2f}")
                st.success("Prediction generated successfully!")
                
            with col2:
                st.subheader("Inputs Provided")
                raw_inputs = {
                    "Age": age,
                    "Sex": sex,
                    "BMI": f"{bmi:.1f}",
                    "Children": children,
                    "Smoker": smoker,
                    "Region": region
                }
                inputs_df = pd.DataFrame.from_dict(
                    raw_inputs, 
                    orient='index', 
                    columns=['Your Input']
                )
                st.dataframe(inputs_df, use_container_width=True)
            
            st.markdown("---") # Add a separator
            
            st.info(
    "**Note:** This is an estimate based on a Linear Regression model and is for informational purposes only. "
    "Actual charges will vary. This is not a substitute for professional medical advice."
)
    else:
        st.error("Could not make prediction. Check file loading errors (files may be missing).")
else:
    # Show a prompt to the user if they haven't clicked the button yet
    st.info("Click the 'Predict Charges' button to generate a prediction.")