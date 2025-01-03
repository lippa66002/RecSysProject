import optuna
from scipy.constants import alpha
import scipy.sparse as sps
import DataHandler
from ModelController import ModelController

from Optimize.SaveResults import SaveResults
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

'''
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
'''



controller = ModelController()

bestrp3 = RP3betaRecommender(controller.URM_train)
bestrp3.load_model(folder_path="_saved_models", file_name="rp3train")
item = ItemKNNCFRecommender(controller.URM_train)
item.load_model(folder_path="_saved_models", file_name="itemtrain")
hyb1 = HybridOptunable2(controller.URM_train)
hyb1.fit(0.5903144712291872,item,bestrp3)
stacked = sps.vstack([0.8718332784366307 * controller.URM_train, (1 - 0.8718332784366307) * controller.ICM_all.T]).tocsr()
slim1 = SLIMElasticNetRecommender(stacked)
slim1.load_model(folder_path="_saved_models", file_name="SLIMstackedTrain2")
slim2 = SLIMElasticNetRecommender(controller.URM_train)
slim2.load_model(folder_path="_saved_models", file_name="SLIMtrain")
hyb2 = HybridOptunable2(controller.URM_train)
hyb2.fit(0.26923209107398516,slim1,slim2)
stacked2 = sps.vstack([0.6814451172353111 * controller.URM_train, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()
slim3 = SLIMElasticNetRecommender(stacked2)
slim3.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval1")
hyb3 = HybridOptunable2(controller.URM_train)
hyb3.fit(0.061079431624077155,hyb2,slim3)
bpr = SLIM_BPR_Cython(controller.URM_train)
bpr.load_model(folder_path="_saved_models", file_name="bprtrain")
hyb5 = HybridOptunable2(controller.URM_train)
hyb5.fit(0.27959722573911727,slim2,slim3)
hyb6 = HybridOptunable2(controller.URM_train)
hyb6.fit(0.18923840370620948,hyb5,bestrp3)
user = UserKNNCFRecommender(controller.URM_train)
user.load_model(folder_path="_saved_models", file_name="Usertrain")
stacked = sps.vstack([0.8392863849420211 * controller.URM_train, (1 - 0.8392863849420211) * controller.ICM_all.T]).tocsr()
slim4 = SLIMElasticNetRecommender(stacked)
slim4.load_model(folder_path="_saved_models",file_name="SLIMstackedTrain3")
rp32 = RP3betaRecommender(stacked)
rp32.fit(topK= 21, beta= 0.2263343041398906, alpha= 0.47403955777118195)
hyb8 = HybridOptunable2(controller.URM_train)
hyb8.fit(0.8095536774911644,rp32,slim4)
itemmm = ItemKNNCFRecommender(stacked)
itemmm.fit(similarity= 'cosine', topK=5, shrink= 0)
hyb9 = HybridOptunable2(controller.URM_train)
hyb9.fit(0.039698998762088614,hyb8,itemmm)
dd,_ = controller.evaluator_test.evaluateRecommender(hyb9)
ease1 = EASE_R_Recommender(controller.URM_train)
ease1.load_model(folder_path="_saved_models", file_name="easetrain")
'''
def objective_function_scores_hybrid_1( optuna_trial):

    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = ScoresHybridRecommender(controller.URM_train, ease1, hyb6, bestrp3, slim1, slim1)

    alpha = optuna_trial.suggest_float("alpha", 0.0, 1.0)



    recom1.fit(alpha, 1-alpha, 0, 0, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]
optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_1,
                              callbacks=[save_results],
                              n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)
'''

def objective_function_scores_hybrid_6( optuna_trial):
    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = HybridOptunable2(controller.URM_train)

    alpha = optuna_trial.suggest_float("alpha", 0, 1)

    recom1.fit(alpha,bestrp3,ease1)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_6,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)




