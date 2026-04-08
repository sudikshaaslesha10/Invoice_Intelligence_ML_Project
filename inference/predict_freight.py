import joblib
import pandas as pd

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "freight_cost_model.pkl")

def load_model(model_path :str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path,"rb") as f:
        model =joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data :dict

    Returns
    ----------
    pd.DataFrame with predicted freight cost
    """
    model=load_model()
    input_df = pd.DataFrame(input_data)

    # 🔥 Automatically match training features
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    input_df['Predicted_freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":

    # Example inference run (local testing)
    sample_data = {
    "Quantity": [100, 200, 150, 50],
    "Dollars": [18500, 9000, 3000, 200]
}
    prediction = predict_freight_cost(sample_data)
    print(prediction)

