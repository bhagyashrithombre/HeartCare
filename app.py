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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "saved_models", "heart_model_v20261013_160903.joblib"
)
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
    "age": 45,
    "sex": 1,
    "cp": 0,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

RANGES = {
    "age": (1, 100),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (80, 220),
    "chol": (100, 600),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (60, 220),
    "exang": (0, 1),
    "oldpeak": (0.0, 6.0),
    "slope": (0, 2),
    "ca": (0, 4),
    "thal": (0, 3)
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
[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] p {
    color: #263D5C;
}
.hero {
    background: linear-gradient(120deg, #142744, #2859A5);
    padding: 26px; border-radius: 16px; margin-bottom: 22px;
}
.hero h1 { color: white !important; font-size: 34px !important; margin: 0 0 8px 0; }
.hero p { color: white !important; margin: 0; font-size: 16px; }
[data-testid="stMetric"] {
    background: white; border: 1px solid #D4E0F0; border-radius: 13px; padding: 17px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #425873 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
    color: #142744 !important; font-weight: 800 !important;
}
label, [data-testid="stWidgetLabel"] p { color: #183153 !important; font-weight: 600 !important; }
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input {
    background: white !important; color: #183153 !important; border-color: #C4D3E7 !important;
}
.stFormSubmitButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
}
.stFormSubmitButton > button p { color: #ffffff !important; }
.stFormSubmitButton > button:hover {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}
.stDownloadButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
    border-radius: 8px !important;
}
.stDownloadButton > button p { color: #ffffff !important; }
.stDownloadButton > button:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
    color: #ffffff !important;
}
.stDownloadButton > button:hover p { color: #ffffff !important; }
[data-testid="stPlotlyChart"] {
    background: white; border: 1px solid #D4E0F0; border-radius: 13px; padding: 8px;
}
.stNumberInput button {
    background-color: #ffffff !important;
    color: #ffffff !important;
    border: 1px solid #C4D3E7 !important;
}
.stNumberInput button svg {
    fill: #142744 !important;
    color: #142744 !important;
}
.stNumberInput button:focus { outline: 2px solid #93c5fd !important; 
}

.stTextArea textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    caret-color: #000000 !important;
}

hr { border-color: #D4E0F0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path):
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
    fig.update_xaxes(
        title_font=dict(color="#183153"),
        tickfont=dict(color="#183153"),
        gridcolor="#E4EAF3"
    )
    fig.update_yaxes(
        title_font=dict(color="#183153"),
        tickfont=dict(color="#183153"),
        gridcolor="#E4EAF3"
    )
    return fig


if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Put heart_model_v20261013_160903.joblib "
        "inside the saved_models folder."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception:
    st.error("Could not load the model. Check that the model file is valid.")
    st.stop()

feature_names = list(getattr(model, "feature_names_in_", FEATURES))

if len(feature_names) != len(FEATURES) or set(feature_names) != set(FEATURES):
    st.error("The model features do not match the HeartCare input fields.")
    st.stop()

if "patient_values" not in st.session_state:
    st.session_state.patient_values = DEFAULTS.copy()

with st.sidebar:
    st.markdown("## HeartCare")
    st.divider()
    st.markdown("### Navigation")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Patient Details", "Usability Feedback"],
        label_visibility="collapsed"
    )

values = st.session_state.patient_values

if page == "Patient Details":
    st.markdown("""
    <div class="hero">
        <h1>Patient Details</h1>
        <p>Change the values and press Update Prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 1, 100, int(values["age"]))
            sex = st.selectbox(
                "Sex",
                [0, 1],
                index=int(values["sex"]),
                format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
            )
            cp = st.selectbox(
                "Chest Pain Type", [0, 1, 2, 3], index=int(values["cp"])
            )
            trestbps = st.number_input(
                "Resting Blood Pressure", 80, 220, int(values["trestbps"])
            )
            chol = st.number_input(
                "Cholesterol", 100, 600, int(values["chol"])
            )

        with col2:
            fbs = st.selectbox(
                "Fasting Blood Sugar", [0, 1], index=int(values["fbs"])
            )
            restecg = st.selectbox(
                "Resting ECG", [0, 1, 2], index=int(values["restecg"])
            )
            thalach = st.number_input(
                "Maximum Heart Rate", 60, 220, int(values["thalach"])
            )
            exang = st.selectbox(
                "Exercise-Induced Angina", [0, 1], index=int(values["exang"])
            )

        with col3:
            oldpeak = st.number_input(
                "Oldpeak", 0.0, 6.0, float(values["oldpeak"]), step=0.1
            )
            slope = st.selectbox(
                "ST Segment Slope", [0, 1, 2], index=int(values["slope"])
            )
            ca = st.selectbox(
                "Major Vessels (CA)", [0, 1, 2, 3, 4], index=int(values["ca"])
            )
            thal = st.selectbox(
                "Thal", [0, 1, 2, 3], index=int(values["thal"])
            )

        submitted = st.form_submit_button(
            "Update Prediction",
            use_container_width=True
        )

    if submitted:
        new_values = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        checked_data, errors, suggestions = validate_input(
            new_values, feature_names
        )

        if errors:
            st.error("Please correct the following input errors.")
            for error in errors:
                st.warning(error)

            if suggestions:
                st.subheader("Suggested Corrections")
                for suggestion in suggestions:
                    st.info(suggestion)
        else:
            try:
                prediction, probabilities = predict_result(model, checked_data)
                st.session_state.patient_values = new_values
                st.session_state.prediction = prediction
                st.session_state.probabilities = probabilities
                st.session_state.prediction_updated = True
                st.success("Patient details validated and prediction updated.")
            except (ValueError, TypeError, AttributeError, IndexError):
                st.error("Unable to update the prediction. Check the model output.")

elif page == "Usability Feedback":
    st.markdown("""
    <div class="hero">
        <h1>Usability Feedback</h1>
        <p>Share a quick note to help improve the application.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("usability_feedback"):
        rating = st.select_slider(
            "How easy was the application to use?",
            options=["Very difficult", "Difficult", "Okay", "Easy", "Very easy"]
        )
        issue = st.selectbox(
            "Did you face any difficulty while using HeartCare?",
            [
                "No difficulties",
                "Entering patient details",
                "Understanding the prediction result",
                "Understanding the graphs",
                "Using the navigation menu",
                "Application loading slowly",
                "Other"
            ]
        )
        suggestion = st.text_area("What would you improve?")
        submitted_feedback = st.form_submit_button("Submit Feedback")

    if submitted_feedback:
        feedback = pd.DataFrame([{
            "Rating": rating,
            "Issue": issue,
            "Suggestion": suggestion
        }])
        st.success("Thank you for your feedback.")
        st.download_button(
            label="Download Feedback",
            data=feedback.to_csv(index=False).encode("utf-8"),
            file_name="heartcare_feedback.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.markdown("""
    <div class="hero">
        <h1>HeartCare</h1>
        <p>Heart Disease Prediction Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    input_data, errors, suggestions = validate_input(values, feature_names)

    if errors:
        st.error(
            "Prediction cannot be generated until the inputs are corrected. "
            "Open Patient Details to review the values."
        )
        for error in errors:
            st.warning(error)

        if suggestions:
            st.subheader("Suggested Corrections")
            for suggestion in suggestions:
                st.info(suggestion)

    else:
        try:
            prediction, probabilities = predict_result(model, input_data)

            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                final_model = list(model.named_steps.values())[-1]
                classes = getattr(final_model, "classes_", None)

            probability = None
            if probabilities is not None:
                if classes is not None and 1 in list(classes):
                    probability = float(probabilities[list(classes).index(1)])
                elif len(probabilities) > 1:
                    probability = float(probabilities[1])

            st.subheader("Prediction Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Patient Age", f"{values['age']} years")
            col2.metric(
                "Prediction",
                "Higher Risk" if prediction == 1 else "Lower Risk"
            )
            col3.metric(
                "Disease Probability",
                f"{probability * 100:.1f}%" if probability is not None else "Not available"
            )

            if prediction == 1:
                st.error(
                    "The model indicates a higher likelihood of heart disease. "
                    "This is not a medical diagnosis."
                )
            else:
                st.success(
                    "The model indicates a lower likelihood of heart disease. "
                    "This is not a medical diagnosis."
                )

            left, right = st.columns(2)

            with left:
                measurement_features = ["age", "trestbps", "chol", "thalach"]
                measurement_data = pd.DataFrame({
                    "Measurement": [LABELS[name] for name in measurement_features],
                    "Value": [values[name] for name in measurement_features]
                })
                fig = px.bar(
                    measurement_data,
                    x="Measurement",
                    y="Value",
                    text="Value",
                    color_discrete_sequence=["#3478D4"]
                )
                fig.update_traces(
                    textposition="outside",
                    textfont=dict(color="#183153", size=12)
                )
                st.plotly_chart(
                    style_chart(fig, "Patient Measurements"),
                    use_container_width=True
                )

            with right:
                if probability is not None:
                    probability_data = pd.DataFrame({
                        "Result": ["Lower Risk", "Higher Risk"],
                        "Probability": [
                            (1 - probability) * 100,
                            probability * 100
                        ]
                    })
                    fig = px.bar(
                        probability_data,
                        x="Result",
                        y="Probability",
                        text="Probability",
                        color="Result",
                        color_discrete_map={
                            "Lower Risk": "#269B78",
                            "Higher Risk": "#E45B64"
                        }
                    )
                    fig.update_traces(
                        texttemplate="%{text:.1f}%",
                        textposition="outside",
                        textfont=dict(color="#183153", size=12)
                    )
                    fig.update_yaxes(title="Probability (%)", range=[0, 115])
                    st.plotly_chart(
                        style_chart(fig, "Prediction Probability"),
                        use_container_width=True
                    )
                else:
                    st.info("Prediction probabilities are not available for this model.")

            st.divider()
            st.subheader("Input Visualization")

            graph_data = pd.DataFrame({
                "Feature": [LABELS[name] for name in feature_names],
                "Value": [values[name] for name in feature_names],
                "Input Range (%)": [
                    (float(values[name]) - RANGES[name][0]) /
                    (RANGES[name][1] - RANGES[name][0]) * 100
                    for name in feature_names
                ]
            })

            fig = px.bar(
                graph_data,
                x="Input Range (%)",
                y="Feature",
                orientation="h",
                hover_data=["Value"],
                range_x=[0, 100],
                color_discrete_sequence=["#3478D4"]
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(
                style_chart(fig, "Patient Input Values", height=500),
                use_container_width=True
            )

        except (ValueError, TypeError, AttributeError, IndexError) as error:
            st.error(
                "Prediction failed because the input or model output is invalid."
            )
            st.caption(str(error))
