# This is a sample Python script.
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import pandas as pd
import scipy.sparse as sps

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


URM_all_dataframe = pd.read_csv(filepath_or_buffer="Data/data_train.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')
users = pd.read_csv(filepath_or_buffer="Data/data_target_users_test.csv")

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
URM_all.tocsr()
recom = ItemKNNCFRecommender(URM_all)
recom.fit(shrink=10, topK=10)
cutoff = 10  # Numero di raccomandazioni da generare
recommendations_list = []
for user_id in users["user_id"]:
    recommendations = recom.recommend(user_id, cutoff=cutoff)
    recommendations_list.append([user_id, recommendations])
df_recommendations = pd.DataFrame(recommendations_list, columns=['user_id', 'item_list'])
df_recommendations.to_csv('recomm.csv', index=False)
