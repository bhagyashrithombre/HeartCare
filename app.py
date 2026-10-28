import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from input import validate_input, predict_result

st.set_page_config(
    page_title="HeartCare",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication Barrier (Uses Streamlit Cloud Secrets)
def check_authentication():
    if "auth" not in st.secrets:
        return True

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("### 🔒 HeartCare Secure Access")
    pwd = st.text_input("Enter Access Passcode", type="password")
    if st.button("Log In"):
        if pwd == st.secrets["auth"]["password"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials.")
    return False

if not check_authentication():
    st.stop()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "heart_model_v20261013_160903.joblib")
DATA_PATH = os.path.join(BASE_DIR, "heart_disease.csv")

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "Oldpeak",
    "slope": "ST Segment Slope",
    "ca": "Major Vessels",
    "thal": "Thal"
}

DEFAULTS = {
    "age": 45, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
}

RANGES = {
    "age": (1, 100), "sex": (0, 1), "cp": (0, 3), "trestbps": (80, 220),
    "chol": (100, 600), "fbs": (0, 1), "restecg": (0, 2), "thalach": (60, 220),
    "exang": (0, 1), "oldpeak": (0.0, 6.0), "slope": (0, 2), "ca": (0, 4), "thal": (0, 3)
}

st.markdown("""
<style>
.stApp { background: #F3F6FC; color: #183153; }
.block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stHeader"] { background: #F3F6FC; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: #142744 !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}
h1, h2, h3, h4 { color: #183153 !important; font-weight: 800 !important; }
.hero {
    background: linear-gradient(120deg, #142744, #2859A5);
    padding: 26px; border-radius: 16px; margin-bottom: 22px;
}
.hero h1 { color: white !important; font-size: 34px !important; margin: 0 0 8px 0; }
.hero p { color: white !important; margin: 0; font-size: 16px; }
[data-testid="stMetric"] {
    background: white; border: 1px solid #D4E0F0; border-radius: 13px; padding: 17px;
}
.stFormSubmitButton > button, .stDownloadButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
}
[data-testid="stPlotlyChart"] {
    background: white; border: 1px solid #D4E0F0; border-radius: 13px; padding: 8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def style_chart(fig, title, height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=19, color="#142744")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=13, color="#183153"),
        margin=dict(l=20, r=20, t=60, b=45),
        height=height,
        showlegend=False
    )
    fig.update_xaxes(title_font=dict(color="#183153"), tickfont=dict(color="#183153"), gridcolor="#E4EAF3")
    fig.update_yaxes(title_font=dict(color="#183153"), tickfont=dict(color="#183153"), gridcolor="#E4EAF3")
    return fig

model = load_model(MODEL_PATH)
if model is None:
    st.error("Model file not found. Ensure `saved_models/heart_model_v20261013_160903.joblib` is committed to GitHub.")
    st.stop()

feature_names = list(getattr(model, "feature_names_in_", FEATURES))

if "patient_values" not in st.session_state:
    st.session_state.patient_values = DEFAULTS.copy()

with st.sidebar:
    st.markdown("## HeartCare")
    st.divider()
    page = st.radio("Navigation", ["Dashboard", "Patient Details", "Usability Feedback"])
    if "auth" in st.secrets:
        st.divider()
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.rerun()

values = st.session_state.patient_values

if page == "Patient Details":
    st.markdown("""
    <div class="hero">
        <h1>Patient Details</h1>
        <p>Update indicators to recalculate disease probability.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 1, 100, int(values["age"]))
            sex = st.selectbox("Sex", [0, 1], index=int(values["sex"]), format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=int(values["cp"]))
            trestbps = st.number_input("Resting Blood Pressure", 80, 220, int(values["trestbps"]))
            chol = st.number_input("Cholesterol", 100, 600, int(values["chol"]))
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar", [0, 1], index=int(values["fbs"]))
            restecg = st.selectbox("Resting ECG", [0, 1, 2], index=int(values["restecg"]))
            thalach = st.number_input("Maximum Heart Rate", 60, 220, int(values["thalach"]))
            exang = st.selectbox("Exercise-Induced Angina", [0, 1], index=int(values["exang"]))
        with col3:
            oldpeak = st.number_input("Oldpeak", 0.0, 6.0, float(values["oldpeak"]), step=0.1)
            slope = st.selectbox("ST Segment Slope", [0, 1, 2], index=int(values["slope"]))
            ca = st.selectbox("Major Vessels (CA)", [0, 1, 2, 3, 4], index=int(values["ca"]))
            thal = st.selectbox("Thal", [0, 1, 2, 3], index=int(values["thal"]))

        submitted = st.form_submit_button("Update Prediction", use_container_width=True)

    if submitted:
        new_values = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        checked_data, errors, suggestions = validate_input(new_values, feature_names)

        if errors:
            st.error("Please correct the following input errors.")
            for error in errors:
                st.warning(error)
        else:
            try:
                prediction, probabilities = predict_result(model, checked_data)
                st.session_state.patient_values = new_values
                st.session_state.prediction = prediction
                st.session_state.probabilities = probabilities
                st.success("Patient indicators validated. Returning to dashboard...")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "Usability Feedback":
    st.markdown("""
    <div class="hero">
        <h1>Usability Feedback</h1>
        <p>Telemetry and feedback collection.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("usability_feedback"):
        rating = st.select_slider("How easy was the application to use?", options=["Very difficult", "Difficult", "Okay", "Easy", "Very easy"])
        issue = st.selectbox("Did you encounter any issues?", ["No difficulties", "Entering details", "Interpreting results", "Slow load time", "Other"])
        suggestion = st.text_area("Feedback / Feature Requests")
        submitted_feedback = st.form_submit_button("Submit Feedback")

    if submitted_feedback:
        feedback = pd.DataFrame([{"Rating": rating, "Issue": issue, "Suggestion": suggestion}])
        st.success("Feedback recorded.")
        st.download_button(
            label="Download Log (CSV)",
            data=feedback.to_csv(index=False).encode("utf-8"),
            file_name="heartcare_feedback.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.markdown("""
    <div class="hero">
        <h1>HeartCare Analytics</h1>
        <p>Machine Learning Heart Disease Diagnostic Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    input_data, errors, suggestions = validate_input(values, feature_names)

    if errors:
        st.error("Input calibration required. Review Patient Details.")
    else:
        prediction, probabilities = predict_result(model, input_data)
        probability = float(probabilities[1]) if probabilities is not None and len(probabilities) > 1 else None

        col1, col2, col3 = st.columns(3)
        col1.metric("Patient Age", f"{values['age']} yrs")
        col2.metric("Risk Status", "Higher Risk" if prediction == 1 else "Lower Risk")
        col3.metric("Probability Index", f"{probability * 100:.1f}%" if probability is not None else "N/A")

        left, right = st.columns(2)
        with left:
            measurement_features = ["age", "trestbps", "chol", "thalach"]
            measurement_data = pd.DataFrame({
                "Measurement": [LABELS[name] for name in measurement_features],
                "Value": [values[name] for name in measurement_features]
            })
            fig = px.bar(measurement_data, x="Measurement", y="Value", text="Value", color_discrete_sequence=["#3478D4"])
            fig.update_traces(textposition="outside")
            st.plotly_chart(style_chart(fig, "Key Clinical Vitals"), use_container_width=True)

        with right:
            if probability is not None:
                prob_df = pd.DataFrame({
                    "Result": ["Lower Risk", "Higher Risk"],
                    "Probability": [(1 - probability) * 100, probability * 100]
                })
                fig_prob = px.bar(prob_df, x="Result", y="Probability", text="Probability", color="Result",
                                  color_discrete_map={"Lower Risk": "#269B78", "Higher Risk": "#E45B64"})
                fig_prob.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_prob.update_yaxes(range=[0, 115])
                st.plotly_chart(style_chart(fig_prob, "Risk Probability Analysis"), use_container_width=True)