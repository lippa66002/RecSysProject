from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
import optuna

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

controller = ModelController()
rp3 = RP3betaRecommender(controller.URM_train)
rp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)
bpr= SLIM_BPR_Cython(controller.URM_train)
bpr.fit(topK= 20, learning_rate= 0.07235759859199255, lambda_i= 0.0026131161353345695, lambda_j= 0.0025413226707704894, symmetric= True, sgd_mode= 'adagrad')
item = ItemKNNCFRecommender(controller.URM_train)
item.fit(similarity= 'tversky', topK= 5, shrink= 15, tversky_alpha= 0.0291003114865242, tversky_beta= 1.0501107741561788)
user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 1000, shrink= 16, similarity ='cosine', normalize= True, feature_weighting= 'BM25')



def objective_function_scores_hybrid_1( optuna_trial):
    print("item + user")

    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = ScoresHybridRecommender(controller.URM_train, rp3, user, rp3, rp3, rp3)

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
    print("item + bpr")
    recom1.fit(alpha,item,bpr)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_6,
                      callbacks=[save_results],
                      n_trials=100)
print(save_results.results_df)
print(optuna_study.best_trial.params)
'''