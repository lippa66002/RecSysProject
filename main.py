
import optuna
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from Optimize.SaveResults import SaveResults
from Optimize.slim import objective_function_SLIM
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import pandas as pd
import scipy.sparse as sps
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_SVDpp_Cython
from Recommenders.MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import Optimize
import time
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender
from prova import cutoff

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


URM_all_dataframe = pd.read_csv(filepath_or_buffer="Data/data_train.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')
users = pd.read_csv(filepath_or_buffer="Data/data_target_users_test.csv")

ICM = pd.read_csv(filepath_or_buffer="Data/data_ICM_metadata.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')




n_interactions = len(URM_all_dataframe)
mapped_id, original_id = pd.factorize(URM_all_dataframe["user_id"].unique())
user_original_ID_to_index = pd.Series(mapped_id, index=original_id)
mapped_id, original_id = pd.factorize(URM_all_dataframe["item_id"].unique())
item_original_ID_to_index = pd.Series(mapped_id, index=original_id)
print(URM_all_dataframe['item_id'].max())
print(URM_all_dataframe.head(4))
#URM_all_dataframe["user_id"] = URM_all_dataframe["user_id"].map(user_original_ID_to_index)
#URM_all_dataframe["item_id"] = URM_all_dataframe["item_id"].map(item_original_ID_to_index)
print(URM_all_dataframe['item_id'].max())
print(URM_all_dataframe.head(4))
userID_unique = URM_all_dataframe["user_id"].unique()
itemID_unique = URM_all_dataframe["item_id"].unique()
n_users = len(userID_unique)
n_items = len(itemID_unique)
print(n_users, n_items)
URM_all = sps.coo_matrix((URM_all_dataframe["data"].values,
                          (URM_all_dataframe["user_id"].values, URM_all_dataframe["item_id"].values)))
n_items = ICM["item_id"].max() + 1
n_features = ICM["feature_id"].max() + 1

ICM_all = sps.coo_matrix((ICM["data"].values, (ICM["item_id"].values, ICM["feature_id"].values)) , shape=(n_items, n_features))

URM_all.tocsr()
ICM_all.tocsr()

URM_trainval, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_trainval, train_percentage = 0.8)



evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])
'''
def objective_function_graph(optuna_trial):
    recomm = RP3betaRecommender(URM_trainval)
    full_hyperp = {
                   "topK": optuna_trial.suggest_int("topK", 5, 1000),
                   "beta": optuna_trial.suggest_float("beta", 0, 1),
        "alpha": optuna_trial.suggest_float("alpha",0,1.5)}
    recomm.fit(**full_hyperp)

    result_df, _ = evaluator_validation.evaluateRecommender(recomm)

    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")

save_results = SaveResults()

optuna_study.optimize(objective_function_graph,
                      callbacks=[save_results],
                      n_trials=80)
pruned_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
optuna_study.best_trial
optuna_study.best_trial.params
save_results.results_df






def objective_function_KNN_similarities(optuna_trial):
    recommender_instance = ItemKNN_CFCBF_Hybrid_Recommender(URM_trainval,ICM_all)
    similarity = optuna_trial.suggest_categorical("similarity",
                                                  ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean'])

    full_hyperp = {"similarity": similarity,
                   "topK": optuna_trial.suggest_int("topK", 5, 1000),
                   "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
                   }

    if similarity == "asymmetric":
        full_hyperp["asymmetric_alpha"] = optuna_trial.suggest_float("asymmetric_alpha", 0, 2, log=False)
        full_hyperp["normalize"] = True

    elif similarity == "tversky":
        full_hyperp["tversky_alpha"] = optuna_trial.suggest_float("tversky_alpha", 0, 2, log=False)
        full_hyperp["tversky_beta"] = optuna_trial.suggest_float("tversky_beta", 0, 2, log=False)
        full_hyperp["normalize"] = True

    elif similarity == "euclidean":
        full_hyperp["normalize_avg_row"] = optuna_trial.suggest_categorical("normalize_avg_row", [True, False])
        full_hyperp["similarity_from_distance_mode"] = optuna_trial.suggest_categorical("similarity_from_distance_mode",
                                                                                        ["lin", "log", "exp"])
        full_hyperp["normalize"] = optuna_trial.suggest_categorical("normalize", [True, False])

    recommender_instance.fit(**full_hyperp)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]
optuna_study = optuna.create_study(direction="maximize")

save_results = SaveResults()

optuna_study.optimize(objective_function_KNN_similarities,
                          callbacks=[save_results],
                          n_trials=80)
pruned_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
optuna_study.best_trial
optuna_study.best_trial.params
save_results.results_df



def objective_function_SLIM(optuna_trial):


    recommender_instance = SLIMElasticNetRecommender(URM_trainval)
    full_hyperp = {"alpha": optuna_trial.suggest_float("alpha", 1e-5, 1e-3),
                   "topK": optuna_trial.suggest_int("topK", 5, 1000),
                   "l1_ratio": optuna_trial.suggest_float("l1_ratio", 1e-3, 0.6),
                   }
    recommender_instance.fit(**full_hyperp)
    #epochs = recommender_instance.get_early_stopping_final_epochs_dict()["epochs"]
    #optuna_trial.set_user_attr("epochs", epochs)
    #optuna_trial.set_user_attr("train_time (min)", (time.time() - start_time) / 60)
    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")


save_results = SaveResults()

optuna_study.optimize(objective_function_SLIM,
                      callbacks=[save_results],
                      n_trials=70)
pruned_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
optuna_study.best_trial
optuna_study.best_trial.params
save_results.results_df

def _feature_importance(result_df):
    regr = RandomForestRegressor(max_depth=2)

    result_df = result_df.copy()
    result_df = (result_df - result_df.mean()) / (result_df.std() + 1e-6)

    y_train = result_df["result"]
    X_train = result_df.copy()
    X_train = X_train.drop(columns=["result"])

    regr.fit(X_train, y_train)

    plt.bar(X_train.columns, regr.feature_importances_)
    plt.show()
results_df = save_results.results_df.copy()
results_df = results_df[results_df["normalize"]==1.0]
results_df = results_df.drop(["normalize"], axis=1)

_feature_importance(results_df)


print("Best trial:")
print("  Value Validation: ", optuna_study.best_trial.value)




'''
recom = SLIMElasticNetRecommender(URM_all)
recom.fit(alpha= 0.0002021210695683939, topK= 856, l1_ratio= 0.23722934371355184)
graphrec = RP3betaRecommender(URM_trainval)
graphrec.fit(topK= 12, alpha= 0.5769111396825488, beta= 0.0019321798490027353)



def  obj_hybrid(optuna_trial):
    print("helloworld")
    alpha = optuna_trial.suggest_float("alpha", 0.1, 0.9)
    recommender_object = HybridOptunable2(URM_trainval)
    recommender_object.fit(alpha,recom,graphrec )
    result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
    return result_df.loc[10][ "MAP"]

optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_SLIM,
                      callbacks=[save_results],
                      n_trials=70)
pruned_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
optuna_study.best_trial
optuna_study.best_trial.params
save_results.results_df



recommendations_list = []
cutoff = 10
for user_id in users["user_id"]:
    recommendations = recom.recommend(user_id, cutoff=cutoff)
    recommendations_list.append([user_id, recommendations])



result_df, _ = evaluator_validation.evaluateRecommender(recom)
print (result_df.loc[10]["MAP"])
df_recommendations = pd.DataFrame(recommendations_list, columns=['user_id', 'item_list'])
df_recommendations.to_csv('recomm.csv', index=False)

