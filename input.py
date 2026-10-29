import numpy as np
import pandas as pd

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

LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
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

CATEGORICAL = {
    "sex", "cp", "fbs", "restecg", "exang",
    "slope", "ca", "thal"
}


def validate_input(values, feature_names):
    errors = []
    suggestions = []
    cleaned_values = {}

    for name in feature_names:
        label = LABELS.get(name, name)

        if name not in values:
            errors.append(f"{label}: value is missing.")
            continue

        value = values[name]

        if value is None or isinstance(value, str) and not value.strip():
            errors.append(f"{label}: please enter a value.")
            continue

        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError):
            errors.append(f"{label}: enter a valid number.")
            continue

        if not np.isfinite(number):
            errors.append(f"{label}: enter a finite number.")
            continue

        minimum, maximum = RANGES[name]

        if number < minimum or number > maximum:
            errors.append(
                f"{label}: {number:g} is outside the allowed "
                f"range of {minimum} to {maximum}."
            )
            suggestions.append(
                f"{label}: enter a value between "
                f"{minimum} and {maximum}."
            )
            continue

        if name in CATEGORICAL and not number.is_integer():
            errors.append(f"{label}: enter a whole-number category.")
            suggestions.append(
                f"{label}: use a valid category from "
                f"{int(minimum)} to {int(maximum)}."
            )
            continue

        cleaned_values[name] = (
            int(number) if name in CATEGORICAL else number
        )

    if errors:
        return None, errors, suggestions

    input_data = pd.DataFrame(
        [[cleaned_values[name] for name in feature_names]],
        columns=feature_names
    )

    return input_data, errors, suggestions


def predict_result(model, input_data):
    prediction = int(np.asarray(model.predict(input_data)).ravel()[0])

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(
            model.predict_proba(input_data)
        )[0]

        if not np.isfinite(probabilities).all():
            raise ValueError("The model returned invalid probabilities.")

        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("The model returned probabilities outside 0–1.")

        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("The model returned invalid probability totals.")

    return prediction, probabilities

