from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import randint, uniform, loguniform

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

param_dist = {
    'n_estimators': randint(200, 1501),
    'max_depth': randint(2, 11),
    'learning_rate': loguniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 21),
    'gamma': uniform(0, 10),
    'reg_alpha': loguniform(0.001, 10),
    'reg_lambda': loguniform(0.1, 20),
    'scale_pos_weight': uniform(1, 9)
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=100,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best CV AUC:", random_search.best_score_)
