from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# ===============================
# Individual model functions
# ===============================
def decision_tree_model():
    return DecisionTreeClassifier()

def random_forest_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def logistic_regression_model():
    return LogisticRegression(max_iter=2000, solver="lbfgs")

def knn_model(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def svm_model(kernel='rbf'):
    return SVC(kernel=kernel, probability=True)

def gradient_boosting_model(n_estimators=100):
    return GradientBoostingClassifier(n_estimators=n_estimators)

def xgboost_model():
    # Faster XGBoost
    return xgb.XGBClassifier(
        n_estimators=50,       # reduced from 100
        max_depth=3,           # shallower trees
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0
    )

def lightgbm_model():
    # Faster LightGBM
    return lgb.LGBMClassifier(
        n_estimators=50,       # reduced from default 100
        num_leaves=31,         # smaller tree
        max_depth=5,
        verbose=-1
    )

def adaboost_model(n_estimators=50):
    return AdaBoostClassifier(n_estimators=n_estimators)

def extra_trees_model(n_estimators=50):
    return ExtraTreesClassifier(n_estimators=n_estimators)

def lda_model():
    return LinearDiscriminantAnalysis()

def qda_model():
    return QuadraticDiscriminantAnalysis()

def mlp_model(hidden_layer_sizes=(50,), max_iter=200):
    # Faster MLP
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)

# ===============================
# All models dictionary for Streamlit
# ===============================
models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "Logistic Regression": logistic_regression_model,
    "K-Nearest Neighbors": knn_model,
    "Support Vector Machine": svm_model,
    "Gradient Boosting": gradient_boosting_model,
    "XGBoost": xgboost_model,
    "LightGBM": lightgbm_model,
    "AdaBoost": adaboost_model,
    "Extra Trees": extra_trees_model,
    "Linear Discriminant Analysis": lda_model,
    "Quadratic Discriminant Analysis": qda_model,
    "MLP Neural Network": mlp_model
}