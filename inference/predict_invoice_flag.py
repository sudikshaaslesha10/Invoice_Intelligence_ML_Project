import joblib
import pandas as pd

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "predict_flag_invoice.pkl")

def load_model(model_path :str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path,"rb") as f:
        model =joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data :dict

    Returns
    ----------
    pd.DataFrame with predicted flag
    """
    model=load_model()
    input_df = pd.DataFrame(input_data)
    

    # 🔥 Prevent feature mismatch
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    return input_df

if __name__ == "__main__":
    
    # Sample input (must match training features)
    input_data = {
        "invoice_quantity": [10],
        "invoice_dollars": [5000],
        "Freight": [200],
        "total_item_quantity": [50],
        "total_item_dollars": [20000]
    }

    result = predict_invoice_flag(input_data)
    print(result)
