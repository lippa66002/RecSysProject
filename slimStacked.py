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


def objective_function_SLIM(optuna_trial):
    # Aggiunta di gamma ai parametri del trial
    gamma = optuna_trial.suggest_float("gamma", 0.75, 1.0)  # Gamma può variare tra 0 e 1

    # Creazione di stacked con gamma
    stacked = sps.vstack([gamma * URM_trainval, (1 - gamma) * ICM_all.T]).tocsr()


    # Creazione dell'istanza del recommender
    recommender_instance = SLIMElasticNetRecommender(stacked)

    # Hyperparametri da ottimizzare


    # Allenamento del recommender
    recommender_instance.fit(alpha=0.00016789623535227514, topK= 758, l1_ratio= 0.06153944206950855)

    # Valutazione del recommender
    result_df, _ = evaluator_test.evaluateRecommender(recommender_instance)

    # Restituzione del MAP

    return result_df.loc[10]["MAP"]



optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_SLIM,
                              callbacks=[save_results],
                              n_trials=50)

print(save_results.results_df)
print(optuna_study.best_trial.params)



