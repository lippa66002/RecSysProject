import numpy as np
import optuna
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from Optimize.SaveResults import SaveResults
from Optimize.slim import objective_function_SLIM
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import pandas as pd
import scipy.sparse as sps
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_SVDpp_Cython
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import Optimize
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender


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
print(n_users, n_items)
URM_all = sps.coo_matrix((URM_all_dataframe["data"].values,
                          (URM_all_dataframe["user_id"].values, URM_all_dataframe["item_id"].values)))
n_items = ICM["item_id"].max() + 1
n_features = ICM["feature_id"].max() + 1
ICM_all = sps.coo_matrix((ICM["data"].values, (ICM["item_id"].values, ICM["feature_id"].values)) , shape=(n_items, n_features))

URM_all.tocsr()
ICM_all.tocsr()

URM_trainval, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_trainval, train_percentage = 0.8)

numgroups = int(35735/20)
user_interactions = np.array(URM_all.sum(axis=1)).flatten()
URM_groups = []
used_users = set()
URM_all = URM_all.tocsr()
print(URM_all.shape)
# Step 2: Ordinare gli utenti in base al numero di interazioni
sorted_users = np.argsort(user_interactions)  # crescente
# Se preferisci decrescente, usa np.argsort(-user_interactions)

# Step 3: Creare i gruppi
for i in range(19):
    # Prendere gli indici degli utenti per il gruppo corrente
    group_users = sorted_users[i * numgroups: (i + 1) * numgroups]
    used_users.update(group_users)

    # Creare una nuova URM per il gruppo corrente
    group_URM = URM_all[group_users, :]
    URM_groups.append(group_URM)

    # Creare l'ultimo gruppo con gli utenti rimanenti
remaining_users = np.setdiff1d(sorted_users, list(used_users))
last_group_URM = URM_all[remaining_users, :]
URM_groups.append(last_group_URM)
first_group_URM = URM_groups[0]

MAP_recommender_per_group = {}

# Configura i modelli per essere fittati in ogni gruppo
recommender_object_dict = {
    "ItemKNN": ItemKNNCFRecommender,
    "UserKNN": UserKNNCFRecommender,
    "SLIMElasticNet": SLIMElasticNetRecommender,
    "RP3beta": RP3betaRecommender,
    "TopPop": TopPop
}

# Parametri per i modelli
recommender_params = {
    "ItemKNN": {"similarity": "cosine", "topK": 8, "shrink": 12},
    "UserKNN": {"similarity": "dice", "topK": 19, "shrink": 737},
    "SLIMElasticNet": {"alpha": 0.0002021210695683939, "topK": 856, "l1_ratio": 0.23722934371355184},
    "RP3beta": {"topK": 12, "alpha": 0.5769111396825488, "beta": 0.0019321798490027353}
}

# Per ogni gruppo
for group_id, group_URM in enumerate(URM_groups):

    # Separare gli utenti del gruppo dagli altri
    users_in_group = group_URM.nonzero()[0]  # Gli utenti in questo gruppo
    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    # Valutatore con utenti esclusi dal gruppo
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], ignore_users=users_not_in_group)

    print(f"Fitting and evaluating models for Group {group_id + 1}")

    # Fittare e valutare ogni raccomandatore per il gruppo corrente
    for label, recommender_class in recommender_object_dict.items():
        # Inizializzare il raccomandatore
        recommender = recommender_class(group_URM)

        # Fittare il modello con i parametri specifici
        if label in recommender_params:
            recommender.fit(**recommender_params[label])
        else:
            recommender.fit()  # Per TopPop

# Valutare il modello
        result_df, _ = evaluator_test.evaluateRecommender(recommender)
        group_map = result_df.loc[10]["MAP"]  # MAP a cutoff 10

        # Salvare i risultati
        if label in MAP_recommender_per_group:
            MAP_recommender_per_group[label].append(group_map)
        else:
            MAP_recommender_per_group[label] = [group_map]

# Creare il grafico
plt.figure(figsize=(16, 9))
for label, results in MAP_recommender_per_group.items():
    plt.scatter(x=np.arange(0, len(results)), y=results, label=label)
plt.ylabel("MAP")
plt.xlabel("User Group")
plt.title("MAP per User Group per Recommender")
plt.legend()
plt.show()