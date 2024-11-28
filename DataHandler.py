import pandas as pd
import scipy.sparse as sps


def create_urm_icm(URM_all_dataframe, ICM):
    n_interactions = len(URM_all_dataframe)
    mapped_id, original_id = pd.factorize(URM_all_dataframe["user_id"].unique())
    user_original_ID_to_index = pd.Series(mapped_id, index=original_id)
    mapped_id, original_id = pd.factorize(URM_all_dataframe["item_id"].unique())
    item_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    userID_unique = URM_all_dataframe["user_id"].unique()
    itemID_unique = URM_all_dataframe["item_id"].unique()
    n_users = len(userID_unique)
    n_items = len(itemID_unique)
    URM_all = sps.coo_matrix((URM_all_dataframe["data"].values,
                              (URM_all_dataframe["user_id"].values, URM_all_dataframe["item_id"].values)))
    n_features = ICM["feature_id"].max() + 1
    ICM_all = sps.coo_matrix((ICM["data"].values, (ICM["item_id"].values, ICM["feature_id"].values)),
                             shape=(n_items, n_features))

    return URM_all.tocsr(), ICM_all.tocsr()