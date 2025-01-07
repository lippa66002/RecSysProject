import scipy.sparse as sps

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
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

slim1 = SLIMElasticNetRecommender(controller.URM_train)
slim1.load_model(folder_path="_saved_models", file_name="SLIMtrain")
stacked = sps.vstack([0.6814451172353111 * controller.URM_boost, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()
slim2 = SLIMElasticNetRecommender(stacked)
slim2.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval1")
bestrp3 = RP3betaRecommender(controller.URM_train)
bestrp3.fit(topK= 12 , alpha = 0.25843, beta= 0.357834)
bestrp3.save_model(folder_path="_saved_models", file_name="rp3_train_f")
hyb1 = HybridOptunable2(controller.URM_train)
hyb1.fit(0.27959722573911727,slim1,slim2)
ease1 = EASE_R_Recommender(controller.URM_train)
ease1.load_model(folder_path="_saved_models", file_name="easetrainll")
hyb2 = HybridOptunable2(controller.URM_train)
hyb2.fit(0.18923840370620948,hyb1,bestrp3)
hyb3 = ScoresHybridRecommender(controller.URM_train, ease1, hyb2, bestrp3, slim1, slim1)
alpha=0.689217356
hyb3.fit(alpha,1-alpha,0,0,0)
user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 995, shrink= 398, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

def objective_function_scores_hybrid_1( optuna_trial):

    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    print("hyb3 + user")
    recom1 = ScoresHybridRecommender(controller.URM_train, ease1, hyb1, user, slim1, slim1)

    alpha = optuna_trial.suggest_float("alpha", 0.66, 0.69)
    gamma = optuna_trial.suggest_float("gamma", 0.0, 0.01)





    recom1.fit(alpha, 1-alpha, gamma, 0, 0)

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