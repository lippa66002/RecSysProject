import optuna
import pandas as pd
import scipy.sparse as sps
import zipfile


from Evaluation.Evaluator import EvaluatorHoldout
from ModelController import ModelController
from Optimize.SaveResults import SaveResults

from Recommenders.Neural.MultVAERecommender import MultVAERecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

controller = ModelController()



def objective_function_multVAE(optuna_trial):
    recommender_instance = MultVAERecommender(controller.URM_train)
    full_hyperp = {
        # "epochs": optuna_trial.suggest_int("epochs", 10, 100),
        "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-7, 1, log=True),
        "batch_size": optuna_trial.suggest_int("batch_size", 1, 2512),
        "dropout": optuna_trial.suggest_float("dropout", 0.1, 0.9),
        "anneal_cap": optuna_trial.suggest_float("anneal_cap", 0.1, 2),
        "epochs" : optuna_trial.suggest_int("epochs" ,50,1000)
    }
    recommender_instance.fit(**full_hyperp)
    result_df, _ = controller.evaluator_test.evaluateRecommender(recommender_instance)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_multVAE,
                              callbacks=[save_results],
                              n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)
