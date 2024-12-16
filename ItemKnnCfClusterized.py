import numpy as np
import optuna
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

import DataHandler
from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import pandas as pd
import scipy.sparse as sps
from Evaluation.Evaluator import EvaluatorHoldout


URM_all_dataframe = pd.read_csv(filepath_or_buffer="Data/data_train.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')
users = pd.read_csv(filepath_or_buffer="Data/data_target_users_test.csv")

ICM = pd.read_csv(filepath_or_buffer="Data/data_ICM_metadata.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')

URM_all, ICM_all = DataHandler.create_urm_icm(URM_all_dataframe, ICM)
controller = ModelController(URM_all, ICM_all)
profile_length = np.ediff1d(sps.csr_matrix(controller.URM_train).indptr)
profile_length, profile_length.shape
block_size = int(len(profile_length)*0.05)
block_size
sorted_users = np.argsort(profile_length)
sorted_users
for group_id in range(0, 20):
    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]

    print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
        group_id,
        users_in_group.shape[0],
        users_in_group_p_len.mean(),
        np.median(users_in_group_p_len),
        users_in_group_p_len.min(),
        users_in_group_p_len.max()))




def objective_function_KNN_similarities(optuna_trial):
    recommender_instance = ItemKNNCFRecommender(controller.URM_train)
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
    evaluator_test = EvaluatorHoldout(controller.URM_test, cutoff_list=[10], ignore_users=users_not_in_group)

    result_df, _ = evaluator_test.evaluateRecommender(recommender_instance)
    return result_df.loc[10]["MAP"]
for group_id in range(15, 20):
    print("Group {}".format(group_id))

    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]

    print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
                group_id,
                users_in_group.shape[0],
                users_in_group_p_len.mean(),
                np.median(users_in_group_p_len),
                users_in_group_p_len.min(),
                users_in_group_p_len.max()))

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    optuna_study = optuna.create_study(direction="maximize")

    save_results = SaveResults()

    optuna_study.optimize(objective_function_KNN_similarities,
                          callbacks=[save_results],
                          n_trials=50)

    print(save_results.results_df)
    print(optuna_study.best_trial.params)
