from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score

def train_random_forest(X_train,y_train):
    rf = RandomForestClassifier(
        random_state =42,
        n_jobs=-1
    )

    param_grid ={"n_estimators":[100,200,300],
             "max_depth":[None,4,5,6],
             "min_samples_split":[2,3,5],
             "min_samples_leaf":[1,2,5],
             "criterion":['entropy','gini']}

    # Scoring metric
    scorer = make_scorer(f1_score)
    
    # GridSearch
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train,y_train)
    return grid_search

def evaluate_classifier(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test,y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy : {round(accuracy, 2)}")
    print(report)