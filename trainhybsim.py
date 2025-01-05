
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
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

controller = ModelController()

stacked = sps.vstack([0.8392863849420211 * controller.URM_train, (1 - 0.8392863849420211) * controller.ICM_all.T]).tocsr()
slim4 = SLIMElasticNetRecommender(stacked)
slim4.fit(alpha =5.632458754549518e-05, topK=619, l1_ratio= 0.053794482642909716)
rp3 = RP3betaRecommender(controller.URM_train)
rp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)
user = UserKNNCFRecommender(controller.URM_train)
user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 1000, shrink= 16, similarity ='cosine', normalize= True, feature_weighting= 'BM25')
bpr= SLIM_BPR_Cython(controller.URM_train)
bpr.fit(topK= 20, learning_rate= 0.07235759859199255, lambda_i= 0.0026131161353345695, lambda_j= 0.0025413226707704894, symmetric= True, sgd_mode= 'adagrad')

def objective_function_scores_hybrid_1( optuna_trial):

    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = ScoresHybridRecommender(controller.URM_train, user, bpr, slim4, rp3, rp3)

    alpha = optuna_trial.suggest_float("alpha", 0.0, 100.0)
    beta = optuna_trial.suggest_float("beta", 0.0, 100.0)
    gamma = optuna_trial.suggest_float("gamma", 0.0, 100.0)
    delta = optuna_trial.suggest_float("delta", 0.0, 100.0)









    recom1.fit(alpha,beta, gamma, delta, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]
optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_1,
                              callbacks=[save_results],
                              n_trials=300)
print(save_results.results_df)
print(optuna_study.best_trial.params)

