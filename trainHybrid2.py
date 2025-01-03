import optuna
import scipy.sparse as sps
from ModelController import ModelController

from Optimize.SaveResults import SaveResults
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

controller = ModelController()

stacked = sps.vstack([0.8718332784366307 * controller.URM_train, (1 - 0.8718332784366307) * controller.ICM_all.T]).tocsr()
stacked2 = sps.vstack([0.6814451172353111 * controller.URM_train, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()
stacked3 = sps.vstack([0.8392863849420211 * controller.URM_train, (1 - 0.8392863849420211) * controller.ICM_all.T]).tocsr()

item = ItemKNNCFRecommender(controller.URM_train)
item.load_model(folder_path="_saved_models", file_name="itemtrain")

itemmm = ItemKNNCFRecommender(stacked3)
itemmm.fit(similarity= 'cosine', topK=5, shrink= 0)

user = UserKNNCFRecommender(controller.URM_train)
user.load_model(folder_path="_saved_models", file_name="Usertrain")

slim1 = SLIMElasticNetRecommender(stacked)
slim1.load_model(folder_path="_saved_models", file_name="SLIMstackedTrain2")

slim2 = SLIMElasticNetRecommender(controller.URM_train)
slim2.load_model(folder_path="_saved_models", file_name="SLIMtrain")

slim3 = SLIMElasticNetRecommender(stacked2)
slim3.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval1")

slim4 = SLIMElasticNetRecommender(stacked3)
slim4.load_model(folder_path="_saved_models",file_name="SLIMstackedTrain3")

bestrp3 = RP3betaRecommender(controller.URM_train)
bestrp3.load_model(folder_path="_saved_models", file_name="rp3train")

rp32 = RP3betaRecommender(stacked3)
rp32.fit(topK= 21, beta= 0.2263343041398906, alpha= 0.47403955777118195)

bpr = SLIM_BPR_Cython(controller.URM_train)
bpr.load_model(folder_path="_saved_models", file_name="bprtrain")

ease1 = EASE_R_Recommender(controller.URM_train)
ease1.load_model(folder_path="_saved_models", file_name="easetrain")

#hyb1 = HybridOptunable2(controller.URM_train)
#hyb1.fit(0.5903144712291872,item,bestrp3)
#hyb2 = HybridOptunable2(controller.URM_train)
#hyb2.fit(0.26923209107398516,slim1,slim2)
#hyb3 = HybridOptunable2(controller.URM_train)
#hyb3.fit(0.061079431624077155,hyb2,slim3)
hyb5 = HybridOptunable2(controller.URM_train)
hyb5.fit(0.27959722573911727,slim2,slim3)
#hyb6 = HybridOptunable2(controller.URM_train)
#hyb6.fit(0.18923840370620948,hyb5,bestrp3)
#hyb8 = HybridOptunable2(controller.URM_train)
#hyb8.fit(0.8095536774911644,rp32,slim4)
#hyb9 = HybridOptunable2(controller.URM_train)
#hyb9.fit(0.039698998762088614,hyb8,itemmm)
dd,_ = controller.evaluator_test.evaluateRecommender(hyb5)
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

def objective_function_scores_hybrid_6(optuna_trial):
    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = HybridOptunable2(controller.URM_train)

    alpha = optuna_trial.suggest_float("alpha", 0, 1)

    recom1.fit(alpha, hyb5 , user)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_6,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)




