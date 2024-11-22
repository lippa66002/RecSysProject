# Earlystopping hyperparameters available in the framework
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import time
def objective_function_SLIM(optuna_trial, URM_train, evaluator_validation):

    full_hyperp = {"validation_every_n": 5,
                "stop_on_validation": True,
                "evaluator_object": evaluator_validation,
                "lower_validations_allowed": 5,  # Higher values will result in a more "patient" earlystopping
                "validation_metric": "MAP",

                # MAX number of epochs (usually 500)
                "epochs": 500,
                 }

    start_time = time.time()
    recommender_instance = SLIM_BPR_Cython(URM_train)
    recommender_instance.fit(topK = optuna_trial.suggest_int("topK", 1,500),
                             lambda_i = optuna_trial.suggest_categorical("lambda_i", 0.001,1),
                             lambda_j = optuna_trial.suggest_categorical("lambda_j", 0.0001, 1),
                             learning_rate = optuna_trial.suggest_float("learning_rate", 1e-4, 1, log=True),
                             ** full_hyperp)
    epochs = recommender_instance.get_early_stopping_final_epochs_dict()["epochs"]
    optuna_trial.set_user_attr("epochs", epochs)
    optuna_trial.set_user_attr("train_time (min)", (time.time() - start_time) / 60)
    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]
