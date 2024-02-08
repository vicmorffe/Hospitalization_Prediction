import pickle
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify

input_file = "hospitalization-RFC_model.bin"

with open(input_file, "rb") as f_in:
    transformers, model = pickle.load(f_in)


def predict_patient(patient_dict, transformers, model):
    """
    Predicts the hospitalization probability for a given patient.

    Parameters:
    - patient_dict (dict): A dictionary containing the patient's information.
    - transformers (dict): A dictionary containing the transformers used for data preprocessing.
    - model: The trained machine learning model.

    Returns:
    - hospitalization (bool): True if the patient is predicted to be hospitalized, False otherwise.
    - y_patient_pred (float): The predicted probability of hospitalization for the patient.
    """
    
    # From patient_dict
    df_patient = pd.Series(patient_dict).to_frame().T

    imputer = transformers["imputer"]
    scaler = transformers["scaler"]
    transformer = transformers["transformer"]

    ## Now we apply the transformations to the single patient dataframe
    X_patient = imputer.transform(df_patient)

    # Scaling the patient data
    X_patient = scaler.transform(X_patient)

    # Normalizing the patient data
    X_patient = transformer.transform(X_patient)

    y_patient_pred = model.predict_proba(X_patient)[0, 1]
    hospitalization = y_patient_pred >= 0.5

    return hospitalization, y_patient_pred


def revive_nan(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = revive_nan(value)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = revive_nan(value)
    elif data == "":
        return np.nan
    else:
        return data
    

def json_to_dict(patient_json):
    patient_dict = dict(patient_json)
    revive_nan(patient_dict)
    return patient_dict


app = Flask("hospitalization")


@app.route("/predict", methods=["POST"])
def predict():
    patient_json = request.get_json()
    patient_dict = json_to_dict(patient_json)
    
    hospitalization, y_pred = predict_patient(patient_dict, transformers, model)

    print("prediction successful!")

    result = {
        "hospitalization_probability": float(y_pred),
        "hospitalization": bool(hospitalization),
    }

    print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
