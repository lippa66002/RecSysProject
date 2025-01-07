import pandas as pd
import DataHandler
from ModelController import ModelController
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from xgboost import XGBRanker
from tqdm import tqdm
import scipy.sparse as sps
import numpy as np

from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender


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

controller = ModelController()


class XGBoostRerankerRecommender:
    def __init__(self, URM_train, XGB_model, df):
        self.URM_train = URM_train
        self.df = df
        self.XGB_model = XGB_model

    def recommend(self, user_ids, cutoff=10, return_scores=True, remove_seen_flag=True, remove_top_pop_flag=True,
                  remove_custom_items_flag=False):
        recommendations = []
        for user_id in user_ids:
            df_slice = self.df[self.df['UserID'] == user_id]
            items = df_slice.ItemID.to_numpy()
            preds = self.XGB_model.predict(df_slice)
            recommendations.append(items[np.argsort(preds)[-cutoff:][::-1]].tolist())

        if return_scores:
            rec, scores = hyb2.recommend(user_ids, cutoff=cutoff, return_scores=return_scores)
            # useless scores
            return np.array(recommendations), scores
        return np.array(recommendations)

    def get_URM_train(self):
        return self.URM_train
stacked = sps.vstack([0.6814451172353111 * controller.URM_train, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()
slim1 = SLIMElasticNetRecommender(stacked)
slim1.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval1")
slim2 = SLIMElasticNetRecommender(controller.URM_train)
slim2.load_model(folder_path="_saved_models", file_name="SLIMtrain")  #0.3588980025585651 slim1slim2 score,  0.7228086650480543 slim1slim2 optunable2
bestrp3 = RP3betaRecommender(controller.URM_train)
bestrp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)
itemmm = ItemKNNCFRecommender(controller.URM_train)
itemmm.fit(similarity= 'tversky', topK= 5, shrink= 15, tversky_alpha= 0.0291003114865242, tversky_beta= 1.0501107741561788)
hyb1 = HybridOptunable2(controller.URM_train)
hyb1.fit(0.7228086650480543,slim1,slim2)    #0.24821268574498187 hyb1bestrp3 hybridoptunable
hyb2 = HybridOptunable2(controller.URM_train)
hyb2.fit(0.24821268574498187,hyb1,bestrp3)

user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 995, shrink= 398, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')
p3 = P3alphaRecommender(controller.URM_train)
p3.fit(topK= 15, alpha= 0.5657433667229401, min_rating= 0, implicit= False, normalize_similarity= True)

recom = ScoresHybridRecommender(controller.URM_train, hyb2, user, p3, p3, p3)

x = 0.9809789503691551
y = 0.3078230973689968

alpha = x
beta = y * (1 - x)
gamma = (1 - x) * (1 - y)

recom.fit(alpha, beta,gamma,0,0)



n_users, n_items = controller.URM_train.shape

finaltrain = pd.DataFrame(index=range(0,n_users), columns = ["ItemID"])
finaltrain.index.name='UserID'


cutoff = 30

for user_id in tqdm(range(n_users)):
    recommendations = hyb2.recommend(user_id, cutoff = cutoff)
    finaltrain.loc[user_id, "ItemID"] = recommendations

finaltrain = finaltrain.explode("ItemID")




URM_test_coo = sps.coo_matrix(controller.URM_test)

correct_recommendations = pd.DataFrame({"UserID": URM_test_coo.row,
                                        "ItemID": URM_test_coo.col})




correct_recommendations
#%%
finaltrain = pd.merge(finaltrain, correct_recommendations, on=['UserID','ItemID'], how='left', indicator='Exist')



finaltrain["Label"] = finaltrain["Exist"] == "both"
finaltrain.drop(columns = ['Exist'], inplace=True)




top_pop = TopPop(controller.URM_train)
top_pop.fit()


item_cf = ItemKNNCFRecommender(controller.URM_train)
item_cf.fit(similarity= 'tversky', topK= 5, shrink= 15, tversky_alpha= 0.0291003114865242, tversky_beta= 1.0501107741561788)


item_cbf = ItemKNNCBFRecommender(controller.URM_train, controller.ICM_all)
item_cbf.fit()

user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 1000, shrink= 16, similarity ='cosine', normalize= True, feature_weighting= 'BM25')

rp3beta = RP3betaRecommender(controller.URM_train)
rp3beta.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)

slimbpr = SLIM_BPR_Cython(controller.URM_train)
slimbpr.fit( topK= 20, learning_rate= 0.07235759859199255, lambda_i= 0.0026131161353345695, lambda_j= 0.0025413226707704894, symmetric= True, sgd_mode= 'adagrad'
)


other_algorithms_tr = {
    "top_pop": top_pop,
    "item_cf": item_cf,
    "item_cbf": item_cbf,
    "rp3beta": rp3beta,
    "SLIM_BPR": slimbpr,
    "user" : user
}




finaltrain = finaltrain.set_index('UserID')

for user_id in tqdm(range(n_users)):
    for rec_label, rec_instance in other_algorithms_tr.items():

        item_list = finaltrain.loc[user_id, "ItemID"].values.tolist()

        all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute = item_list)

        finaltrain.loc[user_id, rec_label] = all_item_scores[0, item_list]

finaltrain = finaltrain.reset_index()
finaltrain = finaltrain.rename(columns = {"index": "UserID"})




item_popularity = np.ediff1d(sps.csc_matrix(controller.URM_train).indptr)

finaltrain['item_popularity'] = item_popularity[finaltrain["ItemID"].values.astype(int)]

user_popularity = np.ediff1d(sps.csr_matrix(controller.URM_train).indptr)
finaltrain['user_profile_len'] = user_popularity[finaltrain["UserID"].values.astype(int)]


finaltrain = finaltrain.sort_values("UserID").reset_index()
finaltrain.drop(columns = ['index'], inplace=True)



groups = finaltrain.groupby("UserID").size().values




n_estimators = 500
learning_rate = 1e-1
reg_alpha = 1e-1
reg_lambda = 1e-1
max_depth = 5
max_leaves = 32
grow_policy = "depthwise"
objective = "map"  #"pairwise", "ndcg"
booster = "gbtree"
use_user_profile = False
random_seed = None
XGB_model = XGBRanker(
    objective='rank:{}'.format(objective),
    n_estimators = int(n_estimators),
    random_state = random_seed,
    learning_rate = learning_rate,
    reg_alpha = reg_alpha,
    reg_lambda = reg_lambda,
    max_depth = int(max_depth),
    max_leaves = int(max_leaves),
    grow_policy = grow_policy,
    verbosity = 1, # 2 if self.verbose else 0,
    booster = booster,
    # enable_categorical = True
)


y_train = finaltrain["Label"]
X_train = finaltrain.drop(columns=["Label"])
X_train["UserID"] = X_train["UserID"].astype(int)
X_train["ItemID"] = X_train["ItemID"].astype(int)



XGB_model.fit(
    X_train,
    y_train,
    group=groups,
    verbose=True
)


stacked = sps.vstack([0.6814451172353111 * controller.URM_train, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()
slim1 = SLIMElasticNetRecommender(stacked)
slim1.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval1")
slim2 = SLIMElasticNetRecommender(controller.URM_train)
slim2.load_model(folder_path="_saved_models", file_name="SLIMtrain")  #0.3588980025585651 slim1slim2 score,  0.7228086650480543 slim1slim2 optunable2
bestrp3 = RP3betaRecommender(controller.URM_train)
bestrp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)
itemmm = ItemKNNCFRecommender(controller.URM_train)
itemmm.fit(similarity= 'tversky', topK= 5, shrink= 15, tversky_alpha= 0.0291003114865242, tversky_beta= 1.0501107741561788)
hyb1 = HybridOptunable2(controller.URM_train)
hyb1.fit(0.7228086650480543,slim1,slim2)    #0.24821268574498187 hyb1bestrp3 hybridoptunable
hyb2 = HybridOptunable2(controller.URM_train)
hyb2.fit(0.24821268574498187,hyb1,bestrp3)

user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 995, shrink= 398, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')
p3 = P3alphaRecommender(controller.URM_train)
p3.fit(topK= 15, alpha= 0.5657433667229401, min_rating= 0, implicit= False, normalize_similarity= True)

recom = ScoresHybridRecommender(controller.URM_train, hyb2, user, p3, p3, p3)

x = 0.9809789503691551
y = 0.3078230973689968

alpha = x
beta = y * (1 - x)
gamma = (1 - x) * (1 - y)

recom.fit(alpha, beta,gamma,0,0)

n_users, n_items = URM_all.shape

user_recommendations_items = []
user_recommendations_user_id = []

for user_id in tqdm(range(n_users)):
    recommendations = hyb2.recommend(user_id, cutoff=cutoff)

    user_recommendations_items.extend(recommendations)
    user_recommendations_user_id.extend([user_id] * len(recommendations))



finrec = pd.DataFrame({"UserID":user_recommendations_user_id, "ItemID":user_recommendations_items})


top_pop = TopPop(URM_all)
top_pop.fit()


item_cf = ItemKNNCFRecommender(URM_all)
item_cf.fit(similarity= 'tversky', topK= 5, shrink= 15, tversky_alpha= 0.0291003114865242, tversky_beta= 1.0501107741561788)


item_cbf = ItemKNNCBFRecommender(URM_all, controller.ICM_all)
item_cbf.fit()

user = UserKNNCFRecommender(URM_all)
user.fit(topK= 1000, shrink= 16, similarity ='cosine', normalize= True, feature_weighting= 'BM25')

rp3beta = RP3betaRecommender(URM_all)
rp3beta.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)

slimbpr = SLIM_BPR_Cython(URM_all)
slimbpr.fit( topK= 20, learning_rate= 0.07235759859199255, lambda_i= 0.0026131161353345695, lambda_j= 0.0025413226707704894, symmetric= True, sgd_mode= 'adagrad'
)


other_algorithms_tr = {
    "top_pop": top_pop,
    "item_cf": item_cf,
    "item_cbf": item_cbf,
    "rp3beta": rp3beta,
    "SLIM_BPR": slimbpr,
    "user" : user
}



finrec = finrec.set_index('UserID')

for user_id in tqdm(range(n_users)):
    for rec_label, rec_instance in other_algorithms_tr.items():

        item_list = finrec.loc[user_id, "ItemID"].values.tolist()

        all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute = item_list)

        finrec.loc[user_id, rec_label] = all_item_scores[0, item_list]

finrec = finrec.reset_index()
finrec = finrec.rename(columns = {"index": "UserID"})




item_popularity = np.ediff1d(sps.csc_matrix(controller.URM_train).indptr)
finrec['item_popularity'] = item_popularity[finrec["ItemID"].values.astype(int)]

user_popularity = np.ediff1d(sps.csr_matrix(controller.URM_train).indptr)
finrec['user_profile_len'] = user_popularity[finrec["UserID"].values.astype(int)]



finrec = finrec.sort_values("UserID").reset_index()
finrec.drop(columns = ['index'], inplace=True)
finrec["UserID"] = finrec["UserID"].astype(int)
finrec["ItemID"] = finrec["ItemID"].astype(int)




recommender = XGBoostRerankerRecommender(URM_all, XGB_model, finrec)


recommendations = recommender.recommend(users['user_id'] , cutoff=10, return_scores=False)


recommendations = [' '.join([str(item) for item in r]).strip() for r in recommendations]

submission_df = pd.DataFrame({'user_id': users['user_id'].unique(), 'item_list': recommendations})


submission_df.to_csv('recomms.csv', index=False)