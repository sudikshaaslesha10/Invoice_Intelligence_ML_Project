from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth =4):
    model= DecisionTreeRegressor(max_depth =max_depth, random_state =42)
    model.fit(X_train,y_train)
    return model

def train_random_forest(X_train, y_train, max_depth =6):
    model= RandomForestRegressor(max_depth =max_depth,random_state =42)
    model.fit(X_train,y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name:str) -> dict:
    """
    Evaluate regression model and return metrics
    """
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)*100
    
    print(f"\n {model_name} Performance:")
    print(f" mae :{mae :.2f}")
    print(f" mse :{mse :.2f}")
    print(f"RMSE : {round(rmse, 2)}")
    print(f" r2 :{r2:.2f}%")

    return{"model_name": model_name,
    "mae": mae,
    "rmse": rmse,
    "r2":r2}