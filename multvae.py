import optuna
import pandas as pd
import scipy.sparse as sps
import zipfile

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Optimize.SaveResults import SaveResults
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

with zipfile.ZipFile("matrici_sparse.zip", "r") as zipf:
    zipf.extract("URM_trainval.npz")
    zipf.extract("URM_test.npz")
    zipf.extract("URM_train.npz")
    zipf.extract("URM_validation.npz")
    zipf.extract("ICM_all.npz")
    zipf.close()

# Carica la matrice sparse
URM_trainval = sps.load_npz("URM_trainval.npz")
URM_train = sps.load_npz("URM_train.npz")
URM_test = sps.load_npz("URM_test.npz")
URM_validation = sps.load_npz("URM_validation.npz")
ICM_all = sps.load_npz("ICM_all.npz")
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
mult = MultVAERecommender(URM_trainval)
mult.fit()
dd, _ = evaluator_test.evaluateRecommender(mult)
print(dd.loc[10]["MAP"])


def objective_function_multVAE(optuna_trial):
    recommender_instance = MultVAERecommender(URM_trainval)
    full_hyperp = {
        # "epochs": optuna_trial.suggest_int("epochs", 10, 100),
        "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": optuna_trial.suggest_int("batch_size", 32, 512),
        "dropout": optuna_trial.suggest_float("dropout", 0.1, 0.5),
        "total_anneal_steps": optuna_trial.suggest_int("total_anneal_steps", 10000, 200000),
        "anneal_cap": optuna_trial.suggest_float("anneal_cap", 0.1, 0.5),
        "p_dims": [URM_trainval.shape[1], URM_trainval.shape[1]],
        "epochs" : optuna_trial.suggest_int("epochs" ,50,300)
    }
    recommender_instance.fit(**full_hyperp)
    result_df, _ = evaluator_test.evaluateRecommender(recommender_instance)
    return result_df.loc[10]["MAP"]
optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_multVAE,
                              callbacks=[save_results],
                              n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)
