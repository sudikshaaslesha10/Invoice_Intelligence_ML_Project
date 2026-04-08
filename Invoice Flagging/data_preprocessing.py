import sqlite3
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_invoice_data():
    conn = sqlite3.connect(r"C:\Users\Sudiksha Aslesha\MY_PROJECTS\ML Project\inventory.db")

    query = """
    With purchase_agg as (
    SELECT p.PONumber, 
    count(distinct p.Brand) as total_brands,
    sum(p.Quantity) as total_item_quantity,
    sum(p.Dollars) as total_item_dollars,
    avg(julianday(p.ReceivingDate)-julianday(p.PODate)) as avg_receiving_delay
    FROM purchases p
    group by p.PONumber
    ) 

    SELECT
        vi.Quantity as invoice_quantity,
        vi.Dollars as invoice_dollars,
        vi.freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) as days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) as days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay

    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
    ON Vi.PONumber = pa.PONumber

    """

    final_df =pd.read_sql_query(query,conn)
    conn.close()
    return final_df


def create_invoice_risk_label(row):
    # Invoice total mismatch with item_level total
    if (abs(row["invoice_dollars"] - row["total_item_dollars"]) >5):
        return 1

    # Abnormally high receiving delay
    if row["avg_receiving_delay"] >10:
        return 1
    return 0

def apply_labels(final_df):
    final_df["flag_invoice"] = final_df.apply(create_invoice_risk_label,axis=1)
    return final_df

def split_data(final_df, features, target):
    X= final_df[features]
    y= final_df[target]

    return train_test_split(
        X,y, test_size=0.2,random_state=42)

def scale_features(X_train, X_test, scaler_path):
    scaler =StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler, "models/scaler.pkl")
    return X_train_scaled, X_test_scaled