import streamlit as st
import pandas as pd
import joblib
import json
import os
import shap
import matplotlib.pyplot as plt

# --- Formatting & Config ---
st.set_page_config(page_title="Sri Lanka Vehicle Predictor", layout="wide", page_icon="🚘")

# Custom UI Styling
st.markdown("""
    <style>
        .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0rem; text-align: center; }
        .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem; text-align: center; }
        .prediction-box { background-color: #F0FDF4; border: 2px solid #22C55E; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 20px; }
        .prediction-text { color: #166534; font-size: 28px; font-weight: bold; margin: 0; }
        div[data-testid="stSidebar"] { background-color: #F8FAFC; }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
MODELS_DIR = "models"
DATA_DIR = "data"
OUTPUTS_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, 'xgb_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_metadata():
    meta_path = os.path.join(DATA_DIR, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None

def load_metrics():
    metrics_path = os.path.join(OUTPUTS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def load_train_data():
    train_path = os.path.join(DATA_DIR, 'train.csv')
    if os.path.exists(train_path):
        return pd.read_csv(train_path)
    return None

# --- Main App Header ---
st.markdown("<div class='main-header'>🚘 Sri Lanka Vehicle Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A powerful Multi-Column Prediction Dashboard driven by XGBoost Machine Learning</div>", unsafe_allow_html=True)
st.divider()

# Load Requirements
model = load_model()
metadata = load_metadata()
train_data = load_train_data()

if not model or not metadata or train_data is None:
    st.error("⚠️ Data or Models not found. Please run the OOP pipeline first.")
    st.code("python run_pipeline.py --input vehicle_data.csv")
    st.stop()

features = metadata['features']
target_col = metadata['target']
brand_mapping = metadata.get('brand_mapping', {})
model_mapping = metadata.get('model_mapping', {})
brand_mapping_int = {int(k): v for k, v in brand_mapping.items()}
model_mapping_int = {int(k): v for k, v in model_mapping.items()}

# Extract Sub-features
location_cols = [f for f in features if f.startswith('Loc_')]
location_options = [loc.replace('Loc_', '') for loc in location_cols]

# --- Inputs Section Display ---
user_inputs = {}
st.markdown("### 🔍 Configure Vehicle Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🚙 Core Identity")
    # Brand
    if 'Brand_Encoded' in features and brand_mapping_int:
        options = sorted(list(brand_mapping_int.values()))
        selected_brand_str = st.selectbox("Vehicle Brand", options=options, index=options.index("Toyota") if "Toyota" in options else 0)
        selected_code = [k for k, v in brand_mapping_int.items() if v == selected_brand_str][0]
        user_inputs['Brand_Encoded'] = float(selected_code)
    
    # Nested columns for Model and Year to sit horizontally
    nested_col1, nested_col2 = st.columns(2)
    
    with nested_col1:
        # Model
        if 'Model_Encoded' in features and model_mapping_int:
            options = sorted(list(model_mapping_int.values()))
            selected_model_str = st.selectbox("Vehicle Model", options=options, index=0)
            selected_code = [k for k, v in model_mapping_int.items() if v == selected_model_str][0]
            user_inputs['Model_Encoded'] = float(selected_code)
            
    with nested_col2:
        # Year
        if 'Year' in features:
            mean_year = int(train_data['Year'].mean())
            user_inputs['Year'] = st.number_input("Manufacture Year", min_value=1900, max_value=2050, value=mean_year, step=1)

with col2:
    st.markdown("#### ⚙️ Metrics & Condition")
    # Engine/Mileage features
    numeric_features = [f for f in features if f not in ['Brand_Encoded', 'Model_Encoded', 'Year', 'Price_Normalized', 'Mileage_Normalized'] and not f.startswith('Loc_')]
    for feature in numeric_features:
        is_binary = set(train_data[feature].dropna().unique()).issubset({0, 1, 0.0, 1.0})
        mean_val = float(train_data[feature].mean())
        if is_binary:
            user_inputs[feature] = st.selectbox(f"{feature}", options=[0, 1], index=0)
        else:
            user_inputs[feature] = st.number_input(f"{feature}", value=float(mean_val))
            
    # Hidden normalizers
    if 'Price_Normalized' in features: user_inputs['Price_Normalized'] = 0.0
    if 'Mileage_Normalized' in features: user_inputs['Mileage_Normalized'] = 0.0

with col3:
    st.markdown("#### 📍 Location")
    if location_options:
        selected_location = st.selectbox("Registered Location", options=location_options)
    else:
        selected_location = None
        
    for feature in location_cols:
        user_inputs[feature] = 1.0 if feature == f'Loc_{selected_location}' else 0.0

# Convert to dataframe and rigidly enforce feature order
input_df = pd.DataFrame([user_inputs])
input_df = input_df[features]

# Center Button
col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
predict_triggered = col_b2.button("🔮 Generate Price Prediction", type="primary", use_container_width=True)

if predict_triggered:
    prediction = model.predict(input_df)[0]
    
    # Showcase Prediction visually
    st.markdown(f"""
    <div class="prediction-box">
        <p class="prediction-text">Estimated {target_col}: Rs. {prediction:,.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Local Explanation:** See the waterfall chart below to understand exactly why your vehicle was priced this way.")
    
    # Generate Local SHAP
    explainer = shap.TreeExplainer(model)
    shap_values_local = explainer(input_df)
    
    # Create narrower/shorter columns to naturally squeeze the graph
    _, center_col, _ = st.columns([1, 4, 1])
    
    with center_col:
        fig, ax = plt.subplots(figsize=(6, 3))
        # Ensure labels don't get cut off on a smaller layout
        plt.subplots_adjust(left=0.3)
        shap.waterfall_plot(shap_values_local[0], show=False)
        st.pyplot(fig)
        plt.close()

# --- Expandable Model Global Insights ---
with st.expander("📊 View Global Model Performance & Insights"):
    st.write("These metrics and charts apply to the model's overall learned framework, not just the single prediction above.")
    it1, it2, it3 = st.tabs(["Performance Validation", "SHAP Feature Effect", "Overall Importance"])
    
    with it1:
        metrics = load_metrics()
        if metrics:
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE Error", f"{metrics.get('RMSE', 0):,.2f}")
            m2.metric("MAE Error", f"{metrics.get('MAE', 0):,.2f}")
            m3.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
            st.write("*Note: An R² Score closer to 1.0 means the model represents the data perfectly.*")
        else:
            st.warning("Metrics file not found. Ensure the Evaluator class executed properly.")
            
    with it2:
        img_path = os.path.join(PLOTS_DIR, 'shap_summary_plot.png')
        if os.path.exists(img_path):
            st.image(img_path, caption="SHAP Summary Evaluation")
        else:
            st.warning("SHAP Summary Plot missing.")
            
    with it3:
        img2_path = os.path.join(PLOTS_DIR, 'feature_importance_bar.png')
        if os.path.exists(img2_path):
            st.image(img2_path, caption="Average Feature Impact")
        else:
            st.warning("Feature Importance bar plot missing.")
