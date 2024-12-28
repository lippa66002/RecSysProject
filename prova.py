import DataHandler
from ModelController import ModelController
from ModelNames import ModelName
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
import optuna
import pandas as pd
import numpy as np


from Recommenders.HybridCBFandSimilarity import HybridRecommenderCBFandSimilarity
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
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


URM_all, ICM_all = DataHandler.create_urm_icm(URM_all_dataframe, ICM)

controller = ModelController()

"""
optuna_params = controller.optunizer(ModelName.Hybrid_UserKNN_CF_CBF)

recommender_instance = controller.generate_model(ModelName.Hybrid_UserKNN_CF_CBF, optuna_params)
"""

model1 = SLIMElasticNetRecommender(controller.URM_train)
model1.fit(alpha= 0.00022742003969239836, topK= 709, l1_ratio= 0.1488442906776265)

model2 = ItemKNNCBFRecommender(controller.URM_train, ICM_all)
model2.fit(topK= 6, shrink= 693, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

best_map = 0.0
for alpha in np.arange(0.0, 1.0, 0.1):
    recommender_instance = HybridRecommenderCBFandSimilarity(controller.URM_train, ICM_all, model1, model2, alpha=alpha)
    result_df, _ = controller.evaluator_test.evaluateRecommender(recommender_instance)
    print("Alpha: {}, Result: {}".format(alpha, result_df.loc[10]["MAP"]))


result_df, _ = controller.evaluator_test.evaluateRecommender(recommender_instance)
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



