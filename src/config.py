input_path = "../input/"
model_output = "../models/"
test_size = 0.2
mlflow = False
target = 'energy_kwh'

lgb_params = {
    'yg': {'n_estimators': 5000,
           'learning_rate': 0.1,
           'lambda_l2': 3,
           'feature_fraction': 0.7,
           'max_depth': -1,
           'objective': 'fair',
           'metric': 'l1',
           'boosting_type': 'dart',
           'rate_drop': 0.6,
           'skip_drop': 0.7,
           'n_jobs': -1,
           'device_type': 'gpu',
           'tree_learner': 'feature',
           'random_state': 1
    },

    'gj': {'n_estimators': 3000,
           'learning_rate': 0.07,
           'lambda_l2': 3,
           'feature_fraction': 0.5,
           'max_depth': -1,
           'objective': 'fair',
           'metric': 'l1',
           'boosting_type': 'dart',
           'rate_drop': 0.5,
           'skip_drop': 0.7,
           'n_jobs': -1,
           'device_type': 'gpu',
           'tree_learner': 'feature',
           'random_state': 1
    }
}