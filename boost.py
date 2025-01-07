import pandas as pd
import DataHandler
from ModelController import ModelController
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from xgboost import XGBRanker
from tqdm import tqdm
import scipy.sparse as sps
import numpy as np

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
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2


stacked = sps.vstack([0.6814451172353111 * controller.URM_boost, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()
slim1 = SLIMElasticNetRecommender(stacked)
slim1.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval1")
slim2 = SLIMElasticNetRecommender(controller.URM_boost)
slim2.load_model(folder_path="_saved_models", file_name="SLIMtrain")  #0.3588980025585651 slim1slim2 score,  0.7228086650480543 slim1slim2 optunable2
bestrp3 = RP3betaRecommender(controller.URM_boost)
bestrp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)
itemmm = ItemKNNCFRecommender(controller.URM_boost)
itemmm.fit(similarity= 'tversky', topK= 5, shrink= 15, tversky_alpha= 0.0291003114865242, tversky_beta= 1.0501107741561788)
hyb1 = HybridOptunable2(controller.URM_boost)
hyb1.fit(0.7228086650480543,slim1,slim2)    #0.24821268574498187 hyb1bestrp3 hybridoptunable
hyb2 = HybridOptunable2(controller.URM_boost)
hyb2.fit(0.24821268574498187,hyb1,bestrp3)

user = UserKNNCFRecommender(controller.URM_boost)
user.fit(topK= 995, shrink= 398, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')
p3 = P3alphaRecommender(controller.URM_boost)
p3.fit(topK= 15, alpha= 0.5657433667229401, min_rating= 0, implicit= False, normalize_similarity= True)

recom = ScoresHybridRecommender(controller.URM_boost, hyb2, user, p3, p3, p3)

x = 0.9809789503691551
y = 0.3078230973689968

alpha = x
beta = y * (1 - x)
gamma = (1 - x) * (1 - y)

recom.fit(alpha, beta,gamma,0,0)


n_users, n_items = controller.URM_boost.shape

training_dataframe = pd.DataFrame(index=range(0,n_users), columns = ["ItemID"])
training_dataframe.index.name='UserID'
cutoff = 30

for user_id in tqdm(range(n_users)):
    recommendations = recom.recommend(user_id, cutoff = cutoff)
    training_dataframe.loc[user_id, "ItemID"] = recommendations

training_dataframe = training_dataframe.explode("ItemID")

URM_validation_coo = sps.coo_matrix(controller.URM_validation)

correct_recommendations = pd.DataFrame({"UserID": URM_validation_coo.row,
                                        "ItemID": URM_validation_coo.col})


training_dataframe = pd.merge(training_dataframe, correct_recommendations, on=['UserID','ItemID'], how='left', indicator='Exist')
training_dataframe["Label"] = training_dataframe["Exist"] == "both"
training_dataframe.drop(columns = ['Exist'], inplace=True)

from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.NonPersonalizedRecommender import TopPop

top_pop = TopPop(controller.URM_boost)
top_pop.fit()


item_cf = ItemKNNCFRecommender(controller.URM_boost)
item_cf.fit(similarity= 'tversky', topK= 5, shrink= 15, tversky_alpha= 0.0291003114865242, tversky_beta= 1.0501107741561788)


item_cbf = ItemKNNCBFRecommender(controller.URM_boost, controller.ICM_all)
item_cbf.fit()

user = UserKNNCFRecommender(controller.URM_boost)
user.fit(topK= 1000, shrink= 16, similarity ='cosine', normalize= True, feature_weighting= 'BM25')

rp3beta = RP3betaRecommender(controller.URM_boost)
rp3beta.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)

slimbpr = SLIM_BPR_Cython(controller.URM_boost)
slimbpr.fit( topK= 20, learning_rate= 0.07235759859199255, lambda_i= 0.0026131161353345695, lambda_j= 0.0025413226707704894, symmetric= True, sgd_mode= 'adagrad')

ials = IALSRecommender(controller.URM_boost)
ials.load_model(folder_path="_saved_models", file_name="IALStrain")


other_algorithms = {
    "top_pop": top_pop,
    "item_cf": item_cf,
    "item_cbf": item_cbf,
    "rp3beta": rp3beta,
    "SLIM_BPR": slimbpr,
    "user" : user,
    "ials": ials
}

training_dataframe = training_dataframe.set_index('UserID')

for user_id in tqdm(range(n_users)):
    for rec_label, rec_instance in other_algorithms.items():

        item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()

        all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute = item_list)

        training_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list]

training_dataframe = training_dataframe.reset_index()
training_dataframe = training_dataframe.rename(columns = {"index": "UserID"})

item_popularity = np.ediff1d(sps.csc_matrix(controller.URM_boost).indptr)

training_dataframe['item_popularity'] = item_popularity[training_dataframe["ItemID"].values.astype(int)]

user_popularity = np.ediff1d(sps.csr_matrix(controller.URM_boost).indptr)
training_dataframe['user_profile_len'] = user_popularity[training_dataframe["UserID"].values.astype(int)]

training_dataframe = training_dataframe.sort_values("UserID").reset_index()
training_dataframe.drop(columns = ['index'], inplace=True)

groups = training_dataframe.groupby("UserID").size().values

y_train = training_dataframe["Label"]
X_train = training_dataframe.drop(columns=["Label"])
X_train["UserID"] = X_train["UserID"].astype(int)
X_train["ItemID"] = X_train["ItemID"].astype(int)

training_dataframe = training_dataframe.set_index('UserID')

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

user_recommendations_items = []
user_recommendations_user_id = []

for user_id in tqdm(range(n_users)):
    recommendations = recom.recommend(user_id, cutoff=cutoff)

    user_recommendations_items.extend(recommendations)
    user_recommendations_user_id.extend([user_id] * len(recommendations))
test_dataframe = pd.DataFrame({"UserID":user_recommendations_user_id, "ItemID":user_recommendations_items})


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

ials = IALSRecommender(controller.URM_train)
ials.load_model(folder_path="_saved_models", file_name="IALStrain")
other_algorithms_tr = {
    "top_pop": top_pop,
    "item_cf": item_cf,
    "item_cbf": item_cbf,
    "rp3beta": rp3beta,
    "SLIM_BPR": slimbpr,
    "user" : user,
    "ials" : ials
}


test_dataframe = test_dataframe.set_index('UserID')

for user_id in tqdm(range(n_users)):
    for rec_label, rec_instance in other_algorithms_tr.items():

        item_list = test_dataframe.loc[user_id, "ItemID"].values.tolist()

        all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute = item_list)

        test_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list]

test_dataframe = test_dataframe.reset_index()
test_dataframe = test_dataframe.rename(columns = {"index": "UserID"})


item_popularity = np.ediff1d(sps.csc_matrix(controller.URM_train).indptr)
test_dataframe['item_popularity'] = item_popularity[test_dataframe["ItemID"].values.astype(int)]

user_popularity = np.ediff1d(sps.csr_matrix(controller.URM_train).indptr)
test_dataframe['user_profile_len'] = user_popularity[test_dataframe["UserID"].values.astype(int)]



test_dataframe = test_dataframe.sort_values("UserID").reset_index()
test_dataframe.drop(columns = ['index'], inplace=True)
test_dataframe["UserID"] = test_dataframe["UserID"].astype(int)
test_dataframe["ItemID"] = test_dataframe["ItemID"].astype(int)


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

n_estimators = 500
learning_rate = 1e-1
reg_alpha = 1e-1
reg_lambda = 1e-1
max_depth = 5
max_leaves = 32
grow_policy = "depthwise"
objective = "map"  # "pairwise", "ndcg"
booster = "gbtree"
use_user_profile = False
random_seed = None
XGB_model = XGBRanker(
    objective='rank:{}'.format(objective),
    n_estimators=int(n_estimators),
    random_state=random_seed,
    learning_rate=learning_rate,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda,
    max_depth=int(max_depth),
    max_leaves=int(max_leaves),
    grow_policy=grow_policy,
    verbosity=1,  # 2 if self.verbose else 0,
    booster=booster,
    # enable_categorical = True
)
XGB_model.fit(
    X_train,
    y_train,
    group=groups,
    verbose=True
)
recommender = XGBoostRerankerRecommender(controller.URM_train, XGB_model, test_dataframe)
result_df, _ = controller.evaluator_test.evaluateRecommender(recommender)
print(result_df.loc[10, 'MAP'])


def objective_xgboost(trial):
    XGB_model = XGBRanker(
        objective=trial.suggest_categorical('objective', ["rank:map", "rank:pairwise", "rank:ndcg"]),
        n_estimators=trial.suggest_int('n_estimators', 10, 5000, log=True),
        learning_rate=trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-5, 1, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-5, 1, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 20),
        max_leaves=trial.suggest_int('max_leaves', 8, 1024),
        grow_policy=trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        verbosity=2,
        booster='gbtree',
        # tree_method = trial.suggest_categorical('tree_method', ['exact', 'approx', 'hist']),
        tree_method='hist',
        gamma=trial.suggest_float('gamma', 1e-7, 10, log=True),
        min_child_weight=trial.suggest_float('min_child_weight', 1e-7, 1, log=True),
        subsample=trial.suggest_float('subsample', 0.1, 0.9),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.1, 0.9),
        # enable_categorical = True
    )

    XGB_model.fit(
        X_train,
        y_train,
        group=groups,
        verbose=True
    )

    recommender = XGBoostRerankerRecommender(controller.URM_train, XGB_model, test_dataframe)
    result_df, _ = controller.evaluator_test.evaluateRecommender(recommender)

    return result_df.loc[10, 'MAP']


import optuna
from Optimize.SaveResults import SaveResults

save_results = SaveResults()

study = optuna.create_study(direction='maximize')
study.optimize(objective_xgboost, callbacks=[save_results],
               n_trials=50)

print(save_results.results_df)
print(study.best_trial.params)

fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig = optuna.visualization.plot_param_importances(study)
fig.show()
fig = optuna.visualization.plot_slice(study)
fig.show()