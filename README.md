# HeartCare - Heart Disease Prediction

## Overview

HeartCare is a Streamlit-based web application that predicts the likelihood of heart disease using a trained machine learning model. It provides patient input validation, prediction results, probability visualization, and usability feedback.

## Features

- Patient details input
- Input validation and correction suggestions
- Heart disease prediction
- Prediction probability visualization
- Patient measurement charts
- Usability feedback and CSV download
- Interactive dashboard

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Joblib
- Scikit-learn

## Project Structure

```text
HeartCare/
├── app.py
├── input.py
├── requirements.txt
├── README.md
├── heart_disease.csv
└── saved_models/
    └── heart_model_v20261013_160903.joblib
```

## Installation

1. Install Python.
2. Open a terminal in the project folder.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Run the following command:

```bash
streamlit run app.py
```

The application will open in your web browser.

## Model File

Place the trained model in the `saved_models` folder with the filename:

```text
heart_model_v20261013_160903.joblib
```

Ensure that the model is compatible with the input features used by the application.

## Disclaimer

HeartCare is an educational machine learning project. Its predictions are not a medical diagnosis and should not replace professional medical advice.
