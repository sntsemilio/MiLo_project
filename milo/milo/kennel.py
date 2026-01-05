from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Aquí definimos las "razas" (modelos) que MiLo sabe buscar
BREEDS = {
    "random_forest": {
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    },
    "logistic_regression": {
        "model": LogisticRegression,
        "params": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "decision_tree": {
        "model": DecisionTreeClassifier,
        "params": {
            "max_depth": [None, 5, 10, 20],
            "criterion": ["gini", "entropy"]
        }
    }
}