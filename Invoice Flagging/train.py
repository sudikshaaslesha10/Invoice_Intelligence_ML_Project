import joblib
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
import os
from model_evaluation import train_random_forest, evaluate_classifier

FEATURES = ['invoice_quantity','invoice_dollars','Freight',
            'total_item_quantity','total_item_dollars']
TARGET = "flag_invoice"

def main():
    
    # Load Data
    df = load_invoice_data()
    df = apply_labels(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    # Scale
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, 'models/scaler.pkl')

    # Train (✅ FIXED)
    grid_search = train_random_forest(X_train_scaled, y_train)

    # Evaluate
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save model — compressed
    os.makedirs("models", exist_ok=True)
    joblib.dump(grid_search.best_estimator_, 'models/predict_flag_invoice.pkl', compress=('zlib', 6))
    
    # Check size
    size_mb = os.path.getsize('models/predict_flag_invoice.pkl') / 1024 / 1024
    print(f"✅ Model saved — size: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()

