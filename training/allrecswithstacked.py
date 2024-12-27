

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
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

with zipfile.ZipFile("../matrici_sparse.zip", "r") as zipf:
    zipf.extract("URM_trainval.npz")
    zipf.extract("URM_test.npz")
    zipf.extract("URM_train.npz")
    zipf.extract("URM_validation.npz")
    zipf.extract("ICM_all.npz")
    zipf.close()

# Carica la matrice sparse
URM_trainval = sps.load_npz("../URM_trainval.npz")
URM_train = sps.load_npz("../URM_train.npz")
URM_test = sps.load_npz("../URM_test.npz")
URM_validation = sps.load_npz("../URM_validation.npz")
ICM_all = sps.load_npz("../ICM_all.npz")

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

def objective_function_item(optuna_trial):
    # Aggiunta di gamma ai parametri del trial
    gamma = optuna_trial.suggest_float("gamma", 0.5, 1.0)  # Gamma può variare tra 0 e 1

    # Creazione di stacked con gamma
    stacked = sps.vstack([gamma * URM_trainval, (1 - gamma) * ICM_all.T]).tocsr()


    # Creazione dell'istanza del recommender
    recommender_instance = ItemKNNCFRecommender(stacked)
    similarity = optuna_trial.suggest_categorical("similarity",
                                                  ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky',
                                                   'euclidean'])

    full_hyperp = {"similarity": similarity,
                   "topK": optuna_trial.suggest_int("topK", 5, 1000),
                   "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
                   }

    if similarity == "asymmetric":
        full_hyperp["asymmetric_alpha"] = optuna_trial.suggest_float("asymmetric_alpha", 0, 2, log=False)
        full_hyperp["normalize"] = True

    elif similarity == "tversky":
        full_hyperp["tversky_alpha"] = optuna_trial.suggest_float("tversky_alpha", 0, 2, log=False)
        full_hyperp["tversky_beta"] = optuna_trial.suggest_float("tversky_beta", 0, 2, log=False)
        full_hyperp["normalize"] = True

    elif similarity == "euclidean":
        full_hyperp["normalize_avg_row"] = optuna_trial.suggest_categorical("normalize_avg_row", [True, False])
        full_hyperp["similarity_from_distance_mode"] = optuna_trial.suggest_categorical(
            "similarity_from_distance_mode", ["lin", "log", "exp"])
        full_hyperp["normalize"] = optuna_trial.suggest_categorical("normalize", [True, False])

    recommender_instance.fit(**full_hyperp)

    # Valutazione del recommender
    result_df, _ = evaluator_test.evaluateRecommender(recommender_instance)

    # Restituzione del MAP

    return result_df.loc[10]["MAP"]

optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_item,
                              callbacks=[save_results],
                              n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)


def objective_function_SLIM_BPR_Cython(optuna_trial):

    # Aggiunta di gamma ai parametri del trial
    gamma = optuna_trial.suggest_float("gamma", 0.5, 1.0)  # Gamma può variare tra 0 e 1

    # Creazione di stacked con gamma
    stacked = sps.vstack([gamma * URM_trainval, (1 - gamma) * ICM_all.T]).tocsr()
    recommender_instance = SLIM_BPR_Cython(stacked)
    full_hyperp = {
        "topK": optuna_trial.suggest_int("topK", 5, 1000),
        "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "lambda_i": optuna_trial.suggest_float("lambda_i", 1e-4, 1e-2, log=True),
        "lambda_j": optuna_trial.suggest_float("lambda_j", 1e-4, 1e-2, log=True),
        "symmetric": optuna_trial.suggest_categorical("symmetric", [True, False]),
        "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"])
    }
    recommender_instance.fit(**full_hyperp)
    result_df, _ = evaluator_test.evaluateRecommender(recommender_instance)
    return result_df.loc[10]["MAP"]



optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_SLIM_BPR_Cython,
                              callbacks=[save_results],
                              n_trials=50)

print(save_results.results_df)
print(optuna_study.best_trial.params)


def objective_function_rp3beta(optuna_trial):
    # Aggiunta di gamma ai parametri del trial
    gamma = optuna_trial.suggest_float("gamma", 0.5, 1.0)  # Gamma può variare tra 0 e 1

    # Creazione di stacked con gamma
    stacked = sps.vstack([gamma * URM_trainval, (1 - gamma) * ICM_all.T]).tocsr()
    recommender_instance = RP3betaRecommender(stacked)
    full_hyperp = {
        "topK": optuna_trial.suggest_int("topK", 5, 1000),
        "alpha": optuna_trial.suggest_float("alpha", 0.0, 2.0),
        "min_rating": optuna_trial.suggest_int("min_rating", 0, 10),
        "implicit": optuna_trial.suggest_categorical("implicit", [True, False]),
        "normalize_similarity": optuna_trial.suggest_categorical("normalize_similarity", [True, False]),
        "beta" : optuna_trial.suggest_float("beta", 0.1,2)
    }
    recommender_instance.fit(**full_hyperp)
    result_df, _ = evaluator_test.evaluateRecommender(recommender_instance)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_rp3beta,
                      callbacks=[save_results],
                      n_trials=50)

print(save_results.results_df)
print(optuna_study.best_trial.params)