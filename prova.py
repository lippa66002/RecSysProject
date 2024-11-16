# This is a sample Python script.

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


import time

from scipy.sparse import csr_matrix

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_SVDpp_Cython
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


#DATA LOADING

URM = pd.read_csv(filepath_or_buffer="Data/data_train.csv",
                                sep=",",
                                header=0,
                                dtype={0:int, 1:int, 2:float},
                                engine='python')
URM.columns = ["user_id", "item_id", "data"]

users = pd.read_csv(filepath_or_buffer="Data/data_target_users_test.csv",
                    header=0)

ICM = pd.read_csv(filepath_or_buffer="Data/data_ICM_metadata.csv",
                  sep=",",
                  header=0,
                  dtype={0:int, 1:int, 2:float},
                  engine='python')
ICM.columns = ["item_id", "feature_id", "data"]


#DATA PREPROCESSING

#create new sequential indexes for users and items
mapped_id, original_id = pd.factorize(URM["user_id"].unique())
user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

mapped_id, original_id = pd.factorize(URM["item_id"].unique())
item_original_ID_to_index = pd.Series(mapped_id, index=original_id)

URM["user_id"] = URM["user_id"].map(user_original_ID_to_index)
URM["item_id"] = URM["item_id"].map(item_original_ID_to_index)

unique_users = URM["user_id"].unique()
unique_items = URM["item_id"].unique()

num_users = len(unique_users)
num_items = len(unique_items)
num_interactions = len(URM)

print ("Number of items\t {}, Number of users\t {}".format(num_items, num_users))
print ("Max ID items\t {}, Max Id users\t {}\n".format(max(unique_items), max(unique_users)))
print ("Average interactions per user {:.2f}".format(num_interactions/num_users))
print ("Average interactions per item {:.2f}\n".format(num_interactions/num_items))

print ("Sparsity {:.2f} %".format((1-float(num_interactions)/(num_items*num_users))*100))


"""

mapping_user_id = pd.DataFrame({"mapped_user_id": np.arange(num_users), "user_id": unique_users})
mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "item_id": unique_items})

URM = pd.merge(left = URM,
               right = mapping_user_id,
               how = "inner",
               on="user_id")
URM = pd.merge(left = URM,
                right = mapping_item_id,
                how = "inner",
                on="item_id")


# Ottieni gli ID unici degli utenti e rimappa gli ID
mapped_id, original_id = pd.factorize(URM["user_id"].unique())

# Crea un DataFrame per la mappatura degli ID
user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

# Sostituisci gli ID degli utenti nel DataFrame originale con gli ID rimappati
URM["user_id"] = URM["user_id"].map(user_original_ID_to_index)

# Stampa il numero di utenti unici
num_users = len(user_original_ID_to_index)
print(f"Numero di utenti unici: {num_users}")

# Stampa il DataFrame aggiornato
#print(URM.head())
"""
#print(URM)

#DATASET SPLITTING

def dataset_splits(ratings, n_users, n_items, validation_percentage: float, testing_percentage: float):
    seed = 1234

    (user_ids_training, user_ids_test,
     item_ids_training, item_ids_test,
     ratings_training, ratings_test) = train_test_split(ratings.user_id,
                                                        ratings.item_id,
                                                        ratings.data,
                                                        test_size=testing_percentage,
                                                        shuffle=True,
                                                        random_state=seed)

    (user_ids_training, user_ids_validation,
     item_ids_training, item_ids_validation,
     ratings_training, ratings_validation) = train_test_split(user_ids_training,
                                                              item_ids_training,
                                                              ratings_training,
                                                              test_size=validation_percentage,
                                                              )

    urm_train = sp.csr_matrix((ratings_training, (user_ids_training, item_ids_training)),
                              shape=(n_users, n_items))

    urm_validation = sp.csr_matrix((ratings_validation, (user_ids_validation, item_ids_validation)),
                                   shape=(n_users, n_items))

    urm_test = sp.csr_matrix((ratings_test, (user_ids_test, item_ids_test)),
                             shape=(n_users, n_items))

    return urm_train, urm_validation, urm_test

URM_train, URM_validation, URM_test = dataset_splits(URM,
                                                     n_users=num_users,
                                                     n_items=num_items,
                                                     validation_percentage=0.10,
                                                     testing_percentage=0.20)

#SIMILARITY MEASURES

def vector_similarity(urm: sp.csc_matrix, shrink: int):
    item_weights = np.sqrt(
        np.sum(urm.power(2), axis=0)
    ).A.flatten()

    n_items = urm.shape[1]
    urm_t = urm.T
    weights = np.empty(shape=(n_items, n_items))
    for item_id in range(n_items):
        numerator = urm_t.dot(urm[:, item_id]).A.flatten()
        denominator = item_weights[item_id] * item_weights + shrink + 1e-6

        weights[item_id] = numerator / denominator

    np.fill_diagonal(weights, 0.0)
    return weights


#TODO: find the right value for the shrink parameter

vector_weights = vector_similarity(URM_train.tocsc(), 0)


# Collaborative Item KNN
itemKNN_recommender = ItemKNNCFRecommender(URM_train)

itemKNN_recommender.fit(topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting='none', similarity_matrix = vector_weights)

cutoff = 10  # Numero di raccomandazioni da generare
recommendations_list = []
for user_id in users["user_id"]:
    recommendations = itemKNN_recommender.recommend(user_id, cutoff=cutoff)
    recommendations_list.append([user_id, recommendations])

# Evaluation Metrics


# Evaluation Procedure

evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
result, _ = evaluator.evaluateRecommender(itemKNN_recommender)


# Hyperparameter Tuning



#Recommendations
print (result.loc[10]["MAP"])
df_recommendations = pd.DataFrame(recommendations_list, columns=['user_id', 'item_list'])
df_recommendations.to_csv('recomm.csv', index=False)















