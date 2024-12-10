import numpy as np
import optuna
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from Optimize.SaveResults import SaveResults
from Optimize.slim import objective_function_SLIM
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
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

profile_length = np.ediff1d(sps.csr_matrix(URM_trainval).indptr)
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

MAP_recommender_per_group = {}

collaborative_recommender_class = {"TopPop": TopPop,
                                       "UserKNNCF": UserKNNCFRecommender,
                                       "ItemKNNCF": ItemKNNCFRecommender,
                                       "RP3beta": RP3betaRecommender,
                                       "SLIM_BPR": SLIM_BPR_Cython,
                                       "SLIM_ElasticNet": SLIMElasticNetRecommender
                                       }

content_recommender_class = {"ItemKNNCBF": ItemKNNCBFRecommender,
                                 "ItemKNNCFCBF": ItemKNN_CFCBF_Hybrid_Recommender
                                 }

recommender_object_dict = {}

for label, recommender_class in collaborative_recommender_class.items():
    recommender_object = recommender_class(URM_train)
    if recommender_object.RECOMMENDER_NAME == "ItemKNNCFRecommender":
        recommender_object.fit(similarity= "cosine", topK= 8, shrink= 12)
        recommender_object_dict[label] = recommender_object

    elif recommender_object.RECOMMENDER_NAME == "UserKNNCFRecommender":
        recommender_object.fit(similarity= "dice", topK= 19, shrink= 737)
        recommender_object_dict[label] = recommender_object

    elif recommender_object.RECOMMENDER_NAME == "RP3betaRecommender":
        recommender_object.fit(topK= 12, alpha= 0.5769111396825488, beta= 0.0019321798490027353)
        recommender_object_dict[label] = recommender_object

    elif recommender_object.RECOMMENDER_NAME == "SLIM_BPR_Recommender":
        recommender_object.fit(topK= 11, learning_rate= 0.04193849345153912, lambda_i= 0.009876208709609856, lambda_j= 0.00044296738036044263, symmetric= True, sgd_mode= 'adagrad')
        recommender_object_dict[label] = recommender_object

    elif recommender_object.RECOMMENDER_NAME == "SLIMElasticNetRecommender":
        recommender_object.load_model(folder_path="_saved_models", file_name="SLIM_ElasticNetTrain")
        recommender_object_dict[label] = recommender_object

    else:
        recommender_object.fit()
        recommender_object_dict[label] = recommender_object

for label, recommender_class in content_recommender_class.items():
    recommender_object = recommender_class(URM_trainval, ICM_all)
    recommender_object.fit()
    recommender_object_dict[label] = recommender_object

cutoff = 10

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

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

    for label, recommender in recommender_object_dict.items():
        result_df, _ = evaluator_test.evaluateRecommender(recommender)
        if label in MAP_recommender_per_group:
            MAP_recommender_per_group[label].append(result_df.loc[cutoff]["MAP"])
        else:
            MAP_recommender_per_group[label] = [result_df.loc[cutoff]["MAP"]]




_ = plt.figure(figsize=(16, 9))
for label, recommender in recommender_object_dict.items():
    results = MAP_recommender_per_group[label]
    plt.scatter(x=np.arange(0, len(results)), y=results, label=label)
plt.ylabel('MAP')
plt.xlabel('User Group')
plt.legend()
plt.show()