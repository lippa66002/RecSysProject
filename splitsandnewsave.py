import pandas as pd
import scipy.sparse as sps
import zipfile

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from leo import save_results

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



matrici = {
    "URM_trainval": URM_trainval,
    "URM_test": URM_test,
    "URM_train": URM_train,
    "URM_validation": URM_validation,
    "ICM_all": ICM_all
}

# Salvataggio delle matrici sparse in singoli file .npz e compressione in un file ZIP
with zipfile.ZipFile("matrici_sparse.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    for nome, matrice in matrici.items():
        # Salva ogni matrice come file .npz
        filename = f"{nome}.npz"
        sps.save_npz(filename, matrice)
        # Aggiungi il file al file ZIP
        zipf.write(filename)

'''
Estrai il file .npz dal ZIP
with zipfile.ZipFile("URM_trainval.zip", "r") as zipf:
    zipf.extract("URM_trainval.npz")

# Carica la matrice sparse
URM_trainval_loaded = sps.load_npz("URM_trainval.npz")


# Controllo: verifica che la matrice caricata sia identica
print((URM_trainval != URM_trainval_loaded).nnz == 0)  # True se uguali
'''
slim = SLIMElasticNetRecommender(URM_trainval)
slim.fit(alpha =  0.00022742003969239836, topK =  709, l1_ratio =  0.1488442906776265)
slim.save_model(folder_path="_saved_models",file_name = "SLIMtrainval")
bpr = SLIM_BPR_Cython(URM_trainval)
bpr.fit(topK =  11, learning_rate =  0.04193849345153912, lambda_i=  0.009876208709609856, lambda_j= 0.00044296738036044263, symmetric =  True, sgd_mode =  'adagrad')
slim.save_model(folder_path="_saved_models",file_name = "bprtrainval")
item = ItemKNNCFRecommender(URM_trainval)
item.fit(similarity =  "cosine", topK =  8, shrink= 12)
slim.save_model(folder_path="_saved_models",file_name = "itemtrainval")

rp3 = RP3betaRecommender(URM_trainval)
rp3.fit(topK= 12, alpha =  0.5769111396825488, beta= 0.0019321798490027353)
slim.save_model(folder_path="_saved_models",file_name = "rp3trainval")

user = UserKNNCFRecommender(URM_trainval)
user.fit(similarity =  "dice", topK= 19, shrink= 737)
slim.save_model(folder_path="_saved_models",file_name = "usertrainval")

hyb = ItemKNN_CFCBF_Hybrid_Recommender(URM_trainval, ICM_all)
hyb.fit(topK =  6, shrink =  167, similarity =  'asymmetric', normalize =  False, feature_weighting =  'BM25', ICM_weight =  0.375006792830105)
slim.save_model(folder_path="_saved_models",file_name = "hybtrainval")
