from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import scipy.sparse as sps
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
import optuna
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

controller = ModelController()


itemhyb= ItemKNN_CFCBF_Hybrid_Recommender(controller.URM_train,controller.ICM_all)
itemhyb.fit(topK= 6, shrink= 167, similarity= 'asymmetric', normalize= False, feature_weighting= 'BM25', ICM_weight= 0.375006792830105)

slim1 = SLIMElasticNetRecommender(controller.URM_train)
slim1.load_model(folder_path="_saved_models", file_name="slim_train_f")
bestrp3 = RP3betaRecommender(controller.URM_train)
bestrp3.fit(topK= 12 , alpha = 0.25843, beta= 0.357834)
hyb2 = HybridOptunable2(controller.URM_train)
hyb2.fit (0.24002684672441646,slim1,bestrp3)
ease1 = EASE_R_Recommender(controller.URM_train)
ease1.load_model(folder_path="_saved_models", file_name="easetrainll")

item = ItemKNNCBFRecommender(controller.URM_train, controller.ICM_all)
item.fit(topK= 9, shrink= 956, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 1000, shrink= 16, similarity ='cosine', normalize= True, feature_weighting= 'BM25')


def objective_function_scores_hybrid_1( optuna_trial):
    print("hyb2 + bpr")

    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = ScoresHybridRecommender(controller.URM_train, ease1, hyb2, user, itemhyb, user)
    alpha = optuna_trial.suggest_float("alpha", 0.65, 0.68)
    gamma = optuna_trial.suggest_float("gamma", 0.0, 0.007)

    delta = optuna_trial.suggest_float("delta", 0.0, 0.005)

    recom1.fit(alpha, 1-alpha, gamma, delta, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]
optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_1,
                              callbacks=[save_results],
                              n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)

#