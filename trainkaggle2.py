from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
import scipy.sparse as sps
import optuna

from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

controller = ModelController()
bestrp3 = RP3betaRecommender(controller.URM_train)
bestrp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)
stacked = sps.vstack([0.8392863849420211 * controller.URM_train, (1 - 0.8392863849420211) * controller.ICM_all.T]).tocsr()
slim4 = SLIMElasticNetRecommender(stacked)
slim4.load_model(folder_path="_saved_models",file_name="SLIMstackedTrain3")
itemmm = ItemKNNCFRecommender(controller.URM_train)
itemmm.fit(similarity= 'tversky', topK= 5, shrink= 15, tversky_alpha= 0.0291003114865242, tversky_beta= 1.0501107741561788)
user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 1000, shrink= 16, similarity ='cosine', normalize= True, feature_weighting= 'BM25')
slimbpr = SLIM_BPR_Cython(controller.URM_train)
slimbpr.fit( topK= 20, learning_rate= 0.07235759859199255, lambda_i= 0.0026131161353345695, lambda_j= 0.0025413226707704894, symmetric= True, sgd_mode= 'adagrad'
)
top_pop = TopPop(controller.URM_boost)
top_pop.fit()
item = ItemKNNCBFRecommender(controller.URM_train, controller.ICM_all)
item.fit(topK= 9, shrink= 956, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')
ease = EASE_R_Recommender(controller.URM_train)
ease.load_model(folder_path="_saved_models", file_name="easetrain3")


def objective_function_scores_hybrid_1( optuna_trial):
    print("user + slim + ease")

    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = ScoresHybridRecommender(controller.URM_train, user, slim4, ease, slim4, slim4)

    alpha = optuna_trial.suggest_float("alpha", 0.0, 1.0)

    beta = 0.006228402274954442





    recom1.fit(beta, 1-beta , alpha, 0, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]
optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_1,
                              callbacks=[save_results],
                              n_trials=200)
print(save_results.results_df)
print(optuna_study.best_trial.params)



def objective_function_scores_hybrid_6( optuna_trial):
    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    print("score itemcf + slim")

    recom1 = HybridOptunable2(controller.URM_train)

    alpha = optuna_trial.suggest_float("alpha", 0, 1)

    recom1.fit(alpha,slim4,itemmm)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_6,
                      callbacks=[save_results],
                      n_trials=200)
print(save_results.results_df)
print(optuna_study.best_trial.params)
