import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():
    # Paths
    db_path = r"C:\Users\Sudiksha Aslesha\MY_PROJECTS\ML Project\inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # ======================
    # Load data
    # ======================
    df = load_vendor_invoice_data(db_path)

    # ======================
    # Prepare data
    # ======================
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    # ======================
    # Train Models

    # Linear Regression
    lr_model = train_linear_regression(X_train, y_train)

    # Decision Tree
    dt_model = train_decision_tree(X_train, y_train)

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)

    # ======================
    # Evaluate Models
    # ======================
    results = []

    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree Regression"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest Regression"))

    # ======================
    # Save Best Model
    # ======================
    # Assuming Random Forest performs best
    joblib.dump(rf_model, model_dir / "freight_cost_model.pkl")

    print("\n✅ Model saved at:", model_dir / "freight_cost_model.pkl")


if __name__ == "__main__":
    main()

print("Training Features:", X.columns.tolist())
