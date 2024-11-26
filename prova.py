import scipy.sparse as sps
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
import optuna
import pandas as pd

from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

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

userID_unique = URM_all_dataframe["user_id"].unique()
itemID_unique = URM_all_dataframe["item_id"].unique()
n_users = len(userID_unique)
n_items = len(itemID_unique)
URM_all = sps.coo_matrix((URM_all_dataframe["data"].values,
                          (URM_all_dataframe["user_id"].values, URM_all_dataframe["item_id"].values)))
n_features = ICM["feature_id"].max() + 1
ICM_all = sps.coo_matrix((ICM["data"].values, (ICM["item_id"].values, ICM["feature_id"].values)) , shape=(n_items, n_features))

URM_all.tocsr()
ICM_all.tocsr()


URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

class SaveResults(object):

    def __init__(self):
        self.results_df = pd.DataFrame(columns = ["result"])

    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]

        self.results_df = self.results_df.append(hyperparam_dict, ignore_index=True)


def objective_function_SLIMElasticNet(optuna_trial):

    alpha = optuna_trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    l1_ratio = optuna_trial.suggest_float('l1_ratio', 1e-2, 1.0, log=False)
    topK = optuna_trial.suggest_int('topK', 50, 500)

    recommender_instance = SLIMElasticNetRecommender(URM_train)
    recommender_instance.fit(alpha=alpha, l1_ratio=l1_ratio, topK=topK)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)
    return result_df.loc[10]["MAP"]

optuna_study_slimEN = optuna.create_study(direction="maximize")

save_results_slimEN = SaveResults()

optuna_study_slimEN.optimize(objective_function_SLIMElasticNet,   #PUT HERE THE OBJECTIVE FUNCTION U WANT
                      callbacks=[save_results_slimEN],
                      n_trials = 50)

print(optuna_study_slimEN.best_trial.params)

recommender_instance = SLIMElasticNetRecommender(URM_train + URM_validation)
recommender_instance.fit(**optuna_study_slimEN.best_trial.params)

result_df, _ = evaluator_test.evaluateRecommender(recommender_instance)
print(result_df)

cutoff = 10  # Numero di raccomandazioni da generare
recommendations_list = []
for user_id in users["user_id"]:
    recommendations = recommender_instance.recommend(user_id, cutoff=cutoff)
    recommendations_list.append([user_id, recommendations])


#result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)
print (result_df.loc[10]["MAP"])
df_recommendations = pd.DataFrame(recommendations_list, columns=['user_id', 'item_list'])
df_recommendations.to_csv('recomm.csv', index=False)



