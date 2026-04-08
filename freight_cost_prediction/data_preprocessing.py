import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str):
    """
    Load vendor_invoice data from SQLite database
    """
    conn = sqlite3.connect(db_path)
    query = "select * from vendor_invoice"
    df= pd.read_sql_query(query,conn)
    conn.close()
    return df

def prepare_features(df= pd.DataFrame):
    """
    select target and features variable
    """
    X = df[["Dollars"]]
    y = df["Freight"]
    return X,y

def split_data(X,y, test_size=0.2, random_state=42):
    """ Split train test data
    """
    return train_test_split(X,y, test_size= 0.2, random_state =42)