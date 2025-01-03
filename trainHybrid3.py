import pandas as pd
import optuna
import DataHandler
from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

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

slim = SLIMElasticNetRecommender(controller.URM_train)
slim.fit(alpha =  0.00022742003969239836, topK =  709, l1_ratio =  0.1488442906776265)

#bpr = SLIM_BPR_Cython(controller.URM_train)
#bpr.fit(topK =  11, learning_rate =  0.04193849345153912, lambda_i=  0.009876208709609856, lambda_j= 0.00044296738036044263, symmetric =  True, sgd_mode =  'adagrad')

#item = ItemKNNCFRecommender(controller.URM_train)
#item.fit(similarity =  "cosine", topK =  8, shrink= 12)

rp3 = RP3betaRecommender(controller.URM_train)
rp3.fit(topK= 12, alpha =  0.5769111396825488, beta= 0.0019321798490027353)

user = UserKNNCFRecommender(controller.URM_train)
user.fit(similarity =  "dice", topK= 19, shrink= 737)

#items = ItemKNNCBFRecommender(controller.URM_train, controller.ICM_all)
#items.fit(topK= 6, shrink= 693, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

hyb = ItemKNN_CFCBF_Hybrid_Recommender(controller.URM_train, controller.ICM_all)
hyb.fit(topK =  6, shrink =  167, similarity =  'asymmetric', normalize =  False, feature_weighting =  'BM25', ICM_weight =  0.375006792830105)





def objective_function_scores_3models(optuna_trial):
    recom = ScoresHybridRecommender(controller.URM_train, slim, rp3, user, hyb, slim)

    x = optuna_trial.suggest_float("x", 0.0, 1.0)
    y = optuna_trial.suggest_float("y", 0.0, 1.0)

    alpha = x
    beta = y * (1 - x)
    gamma = (1 - x) * (1 - y)

    recom.fit(alpha, beta, gamma, 0, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom)
    return result_df.loc[10]["MAP"]

def objective_function_scores_4models(optuna_trial):
    recom = ScoresHybridRecommender(controller.URM_train, slim, rp3, user, hyb, slim)

    # Sample x, y, and z to calculate weights
    x = optuna_trial.suggest_float("x", 0.0, 1.0)
    y = optuna_trial.suggest_float("y", 0.0, 1.0)
    z = optuna_trial.suggest_float("z", 0.0, 1.0)

    alpha = x
    beta = y * (1 - x)
    gamma = z * (1 - x) * (1 - y)
    delta = (1 - x) * (1 - y) * (1 - z)

    recom.fit(alpha, beta, gamma, delta, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_3models,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)

print("now try 4 models")

optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_4models,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)