import DataHandler
from ModelController import ModelController
from ModelNames import ModelName
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
import optuna
import pandas as pd

from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
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

controller = ModelController(URM_all, ICM_all)


#recommender_instance = HybridOptunable2(controller.URM_all)

#recommender_instance.load_model(folder_path="_saved_models", file_name = "HybridOptunable2 (1)")

slim_bpr = SLIM_BPR_Cython(controller.URM_all)
slim_en = SLIMElasticNetRecommender(controller.URM_all)

slim_bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender")
slim_en.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")

similarity_1 = slim_bpr.W_sparse
similarity_2 = slim_en.W_sparse

recommender_instance = ItemKNNSimilarityHybridRecommender(controller.URM_all, similarity_1, similarity_2)
recommender_instance.fit(topK= 340, alpha= 4.464670628236134e-05)

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



