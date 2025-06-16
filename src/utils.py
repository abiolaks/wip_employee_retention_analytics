import joblib
import os
import sys

# Ensure the custom transformer used in the preprocessing pipeline is
# available when unpickling the joblib artifact. This import is not used
# directly in the code, but joblib requires it to locate the class
# definition.
sys.path.append(os.path.dirname(__file__))
from preprocessing_pipeline import FeatureEngineer  # noqa: F401

# model = joblib.load("../model/employee_attrition_model.pkl")
# preprocessor = joblib.load("../artifacts/preprocessor_pipeline.pkl")
# Dynamically resolve path to the model file
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "employee_attrition_model.pkl"
)
PREPROCESSOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "preprocessor_pipeline.pkl"
)

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict_attrition(df):
    X = preprocessor.transform(df)
    probs = model.predict_proba(X)[:, 1]
    df["Attrition_Probability"] = probs
    df["Risk_Flag"] = df["Attrition_Probability"].apply(
        lambda x: "🔴 High Risk" if x > 0.6 else "🟢 Low Risk"
    )
    df["Risk_Label"] = df["Attrition_Probability"].apply(
        lambda x: "High Risk" if x > 0.6 else "Low Risk"
    )
    return df
