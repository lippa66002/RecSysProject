import scipy.sparse as sps

from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import optuna

from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

controller = ModelController()

stacked = sps.vstack([0.6814451172353111 * controller.URM_train, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()
slim1 = SLIMElasticNetRecommender(stacked)
slim1.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval1")
slim2 = SLIMElasticNetRecommender(controller.URM_train)
slim2.load_model(folder_path="_saved_models", file_name="SLIMtrain")  #0.3588980025585651 slim1slim2 score,  0.7228086650480543 slim1slim2 optunable2
bestrp3 = RP3betaRecommender(controller.URM_train)
bestrp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)
itemmm = ItemKNNCFRecommender(controller.URM_train)
itemmm.fit(similarity= 'tversky', topK= 5, shrink= 15, tversky_alpha= 0.0291003114865242, tversky_beta= 1.0501107741561788)
user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 1000, shrink= 16, similarity ='cosine', normalize= True, feature_weighting= 'BM25')
hyb1 = HybridOptunable2(controller.URM_train)
hyb1.fit(0.7228086650480543,slim1,slim2)    #0.24821268574498187 hyb1bestrp3 hybridoptunable
hyb2 = HybridOptunable2(controller.URM_train)
hyb2.fit(0.24821268574498187,hyb1,bestrp3)    #0.24821268574498187 hyb1bestrp3 hybridoptunable
dd, _ = controller.evaluator_test.evaluateRecommender(hyb2)
hyb3 = HybridOptunable2(controller.URM_train)
hyb3.fit(0.036352595657286435,hyb1,itemmm)    #0.24821268574498187 hyb1bestrp3 hybridoptunable

def objective_function_scores_hybrid_1( optuna_trial):

    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    print("hyb3 + user")
    recom1 = ScoresHybridRecommender(controller.URM_train, hyb3, user, bestrp3, slim1, slim1)

    alpha = optuna_trial.suggest_float("alpha", 0.99, 1.0)





    recom1.fit(alpha, 1-alpha, 0, 0, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]
optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_1,
                              callbacks=[save_results],
                              n_trials=200)
print(save_results.results_df)
print(optuna_study.best_trial.params)

'''
def objective_function_scores_hybrid_6( optuna_trial):
    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    print("hyb2 + item hybridoptunable")
    recom1 = HybridOptunable2(controller.URM_train)

    alpha = optuna_trial.suggest_float("alpha", 0, 1)

    recom1.fit(alpha,hyb2,itemmm)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_6,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)
'''