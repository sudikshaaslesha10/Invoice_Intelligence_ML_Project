import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\Sudiksha Aslesha\MY_PROJECTS\models\freight_cost_model.pkl"

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
    input_df['Predicted_freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":

    # Example inference run (local testing)
    sample_data = {"Dollars": [18500,9000,3000,200]}
    prediction = predict_freight_cost(sample_data)
    print(prediction)

