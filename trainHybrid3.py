import optuna
import pandas as pd
import scipy.sparse as sps

import DataHandler
from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
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
slim.load_model(folder_path="_saved_models", file_name="SLIMtrain")

#bpr = SLIM_BPR_Cython(controller.URM_train)
#bpr.fit(topK =  11, learning_rate =  0.04193849345153912, lambda_i=  0.009876208709609856, lambda_j= 0.00044296738036044263, symmetric =  True, sgd_mode =  'adagrad')

#item = ItemKNNCFRecommender(controller.URM_train)
#item.fit(similarity =  "cosine", topK =  8, shrink= 12)

stacked = sps.vstack([0.8392863849420211 * controller.URM_train, (1 - 0.8392863849420211) * controller.ICM_all.T]).tocsr()

rp3 = RP3betaRecommender(stacked)
rp3.load_model(folder_path="_saved_models", file_name="rp3_stacked3_f")
#rp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)
#rp3.save_model(folder_path="_saved_models", file_name="rp3_train_f")

easeR = EASE_R_Recommender(controller.URM_train)
easeR.load_model(folder_path="_saved_models", file_name="easetrainll")
#easeR.fit(topK= 32, l2_norm= 20.402285200199643, normalize_matrix= False)
#slim.save_model(folder_path="_saved_models", file_name="ease_train_f")

p3 = P3alphaRecommender(controller.URM_train)
p3.load_model(folder_path="_saved_models", file_name="p3alpha_train_f")

user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 995, shrink= 398, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

#user.load_model(folder_path="_saved_models", file_name="user_train_f")

#items = ItemKNNCBFRecommender(controller.URM_train, controller.ICM_all)
#items.fit(topK= 6, shrink= 693, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

# hyb = ItemKNN_CFCBF_Hybrid_Recommender(controller.URM_train, controller.ICM_all)
# hyb.fit(topK =  6, shrink =  167, similarity =  'asymmetric', normalize =  False, feature_weighting =  'BM25', ICM_weight =  0.375006792830105)

def objective_function_scores_5models(optuna_trial):
    recom = ScoresHybridRecommender(controller.URM_train, slim, rp3, easeR, user, p3)

    # Sample x, y, and z to calculate weights
    x = optuna_trial.suggest_float("x", 0.0, 1.0)
    y = optuna_trial.suggest_float("y", 0.0, 1.0)
    z = optuna_trial.suggest_float("z", 0.0, 1.0)
    w = optuna_trial.suggest_float("w", 0.0, 1.0)

    alpha = x
    beta = y * (1 - x)
    gamma = z * (1 - x) * (1 - y)
    delta = w * (1 - x) * (1 - y) * (1 - z)
    epsilon = (1 - x) * (1 - y) * (1 - z) * (1 - w)

    recom.fit(alpha, beta, gamma, delta, epsilon)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom)
    return result_df.loc[10]["MAP"]

optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_5models,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)


"""

def objective_function_scores_3models(optuna_trial):
    recom = ScoresHybridRecommender(controller.URM_train, slim, rp3, user, p3, slim)

    x = optuna_trial.suggest_float("x", 0.0, 1.0)
    y = optuna_trial.suggest_float("y", 0.0, 1.0)

    alpha = x
    beta = y * (1 - x)
    gamma = (1 - x) * (1 - y)

    recom.fit(alpha, beta, gamma, 0, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom)
    return result_df.loc[10]["MAP"]

def objective_function_scores_4models(optuna_trial):
    recom = ScoresHybridRecommender(controller.URM_train, slim, rp3, user, p3, slim)

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
"""