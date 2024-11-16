# This is a sample Python script.
import time

import optuna
from scipy.sparse import csr_matrix

from Optuna.SaveResults import SaveResults
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import pandas as pd
import scipy.sparse as sps
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_SVDpp_Cython
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


URM_all_dataframe = pd.read_csv(filepath_or_buffer="Data/data_train.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')
users = pd.read_csv(filepath_or_buffer="Data/data_target_users_test.csv")

ICM = pd.read_csv(filepath_or_buffer="Data/data_ICM_metadata.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')




n_interactions = len(URM_all_dataframe)
mapped_id, original_id = pd.factorize(URM_all_dataframe["user_id"].unique())
user_original_ID_to_index = pd.Series(mapped_id, index=original_id)
mapped_id, original_id = pd.factorize(URM_all_dataframe["item_id"].unique())
item_original_ID_to_index = pd.Series(mapped_id, index=original_id)
print(URM_all_dataframe['item_id'].max())
print(URM_all_dataframe.head(4))
#URM_all_dataframe["user_id"] = URM_all_dataframe["user_id"].map(user_original_ID_to_index)
#URM_all_dataframe["item_id"] = URM_all_dataframe["item_id"].map(item_original_ID_to_index)
print(URM_all_dataframe['item_id'].max())
print(URM_all_dataframe.head(4))
userID_unique = URM_all_dataframe["user_id"].unique()
itemID_unique = URM_all_dataframe["item_id"].unique()
n_users = len(userID_unique)
n_items = len(itemID_unique)
URM_all = sps.coo_matrix((URM_all_dataframe["data"].values,
                          (URM_all_dataframe["user_id"].values, URM_all_dataframe["item_id"].values)))
n_items = ICM["item_id"].max() + 1
n_features = ICM["feature_id"].max() + 1
ICM_all = sps.coo_matrix((ICM["data"].values, (ICM["item_id"].values, ICM["feature_id"].values)) , shape=(n_items, n_features))

URM_all.tocsr()
ICM_all.tocsr()

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])



'''def objective_function_KNN_similarities(optuna_trial):
    recommender_instance = UserKNNCFRecommender(URM_train)
    similarity = optuna_trial.suggest_categorical("similarity",
                                                  ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean'])

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
        full_hyperp["similarity_from_distance_mode"] = optuna_trial.suggest_categorical("similarity_from_distance_mode",
                                                                                        ["lin", "log", "exp"])
        full_hyperp["normalize"] = optuna_trial.suggest_categorical("normalize", [True, False])

    recommender_instance.fit(**full_hyperp)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")

save_results = SaveResults()

optuna_study.optimize(objective_function_KNN_similarities,
                      callbacks=[save_results],
                      n_trials=50)
pruned_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(optuna_study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
print("  Value Validation: ", optuna_study.best_trial.value)'''


def objective_function_funksvd(optuna_trial):
    # Earlystopping hyperparameters available in the framework
    full_hyperp = {"validation_every_n": 5,
                   "stop_on_validation": True,
                   "evaluator_object": evaluator_validation,
                   "lower_validations_allowed": 5,  # Higher values will result in a more "patient" earlystopping
                   "validation_metric": "MAP",

                   # MAX number of epochs (usually 500)
                   "epochs": 500,
                   }

    start_time = time.time()
    recommender_instance = MatrixFactorization_SVDpp_Cython(URM_train)
    recommender_instance.fit(num_factors=optuna_trial.suggest_int("num_factors", 1, 200),
                             sgd_mode=optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
                             batch_size=optuna_trial.suggest_categorical("batch_size",
                                                                         [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
                             item_reg=optuna_trial.suggest_float("item_reg", 1e-5, 1e-2, log=True),
                             user_reg=optuna_trial.suggest_float("user_reg", 1e-5, 1e-2, log=True),
                             learning_rate=optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                             **full_hyperp)

    # Add the number of epochs selected by earlystopping as a "user attribute" of the optuna trial
    epochs = recommender_instance.get_early_stopping_final_epochs_dict()["epochs"]
    optuna_trial.set_user_attr("epochs", epochs)
    optuna_trial.set_user_attr("train_time (min)", (time.time() - start_time) / 60)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]


# %%
optuna_study = optuna.create_study(direction="maximize")

save_results = SaveResults()

optuna_study.optimize(objective_function_funksvd,
                      callbacks=[save_results],
                      n_trials=50)
optuna_study.best_trial
optuna_study.best_trial.params
save_results.results_df






recom = MatrixFactorization_SVDpp_Cython(URM_train)
recom.fit()
cutoff = 10  # Numero di raccomandazioni da generare
recommendations_list = []
for user_id in users["user_id"]:
    recommendations = recom.recommend(user_id, cutoff=cutoff)
    recommendations_list.append([user_id, recommendations])


result_df, _ = evaluator_validation.evaluateRecommender(recom)
print (result_df.loc[10]["MAP"])
df_recommendations = pd.DataFrame(recommendations_list, columns=['user_id', 'item_list'])
df_recommendations.to_csv('recomm.csv', index=False)

