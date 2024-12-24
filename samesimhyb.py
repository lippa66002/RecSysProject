import optuna
from scipy.constants import alpha

import DataHandler
from ModelController import ModelController
import pandas as pd
from ModelNames import ModelName
from Optimize.SaveResults import SaveResults
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

print("Start")
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender


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


slim = SLIMElasticNetRecommender(controller.URM_train)
bpr = SLIM_BPR_Cython(controller.URM_train)
bpr.fit(topK =  11, learning_rate =  0.04193849345153912, lambda_i=  0.009876208709609856, lambda_j= 0.00044296738036044263, symmetric =  True, sgd_mode =  'adagrad')
slim.fit(alpha =  0.00022742003969239836, topK =  709, l1_ratio =  0.1488442906776265)
item = ItemKNNCFRecommender(controller.URM_train)
item.fit(similarity =  "cosine", topK =  8, shrink= 12)
rp3 = RP3betaRecommender(controller.URM_train)
rp3.fit(topK= 12, alpha =  0.5769111396825488, beta= 0.0019321798490027353)
user = UserKNNCFRecommender(controller.URM_train)
user.fit(similarity =  "dice", topK= 19, shrink= 737)
items = ItemKNNCBFRecommender(controller.URM_train, controller.ICM_all)
items.fit()
hyb = ItemKNN_CFCBF_Hybrid_Recommender(controller.URM_train, controller.ICM_all)
hyb.fit(topK =  6, shrink =  167, similarity =  'asymmetric', normalize =  False, feature_weighting =  'BM25', ICM_weight =  0.375006792830105)


itembpr = ScoresHybridRecommender(controller.URM_train, item, bpr, slim, slim, slim)
itembpr.fit(0.78083761190111, 1-alpha, 0, 0, 0)
rp3bpr = ScoresHybridRecommender(controller.URM_train, rp3, bpr, slim, slim, slim)
rp3bpr.fit(0.9103458465533064,1-alpha, 0, 0, 0)
rp3item = ScoresHybridRecommender(controller.URM_train, rp3, item, slim, slim, slim)
rp3item.fit(0.7592306161113771,1-alpha, 0, 0, 0)
bprhyb = ScoresHybridRecommender(controller.URM_train, bpr, hyb, slim, slim, slim)
bprhyb.fit(0.8240882120610954, 1-alpha, 0, 0, 0)
itemsbpr = ScoresHybridRecommender(controller.URM_train, items, bpr, slim, slim, slim)
itemsbpr.fit(0.2127643996908023, 1-alpha, 0, 0, 0)
rp3items = ScoresHybridRecommender(controller.URM_train, rp3, items, slim, slim, slim)
rp3items.fit(0.9359918689312217, 1-alpha, 0, 0, 0)

cc,_ = controller.evaluator_test.evaluateRecommender(slim)
print(cc.loc[10]["MAP"])
cc,_ = controller.evaluator_test.evaluateRecommender(itemsbpr)
print(cc.loc[10]["MAP"])
cc,_ = controller.evaluator_test.evaluateRecommender(rp3bpr)
print(cc.loc[10]["MAP"])
cc,_ = controller.evaluator_test.evaluateRecommender(rp3item)
print(cc.loc[10]["MAP"])
cc,_ = controller.evaluator_test.evaluateRecommender(bprhyb)
print(cc.loc[10]["MAP"])
cc,_ = controller.evaluator_test.evaluateRecommender(itemsbpr)
print(cc.loc[10]["MAP"])
cc,_ = controller.evaluator_test.evaluateRecommender(rp3items)
print(cc.loc[10]["MAP"])


def objective_function_scores_hybrid_5( optuna_trial):
    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = ScoresHybridRecommender(controller.URM_train, rp3, slim, bpr, slim, slim)

    alpha = optuna_trial.suggest_int("alpha", 1, 20)

    recom1.fit(alpha, 1-alpha, 0, 0, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_5,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)








def objective_function_scores_hybrid_6( optuna_trial):
    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = HybridOptunable2(controller.URM_train)

    alpha = optuna_trial.suggest_int("alpha", 0, 1)

    recom1.fit(alpha,slim,item)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_6,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)

def objective_function_scores_hybrid_7( optuna_trial):
    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = HybridOptunable2(controller.URM_train)

    alpha = optuna_trial.suggest_int("alpha", 0, 1)

    recom1.fit(alpha,slim,bpr)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_7,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)
def objective_function_scores_hybrid_8( optuna_trial):
    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = HybridOptunable2(controller.URM_train)

    alpha = optuna_trial.suggest_int("alpha", 0, 1)

    recom1.fit(alpha,slim,rp3)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_8,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)