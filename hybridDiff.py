import DataHandler
from ModelController import ModelController
from ModelNames import ModelName
import numpy as np
import pandas as pd

from Recommenders.HybridDifferentLossFunctions import DifferentLossScoresHybridRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
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

controller = ModelController()

model1 = SLIMElasticNetRecommender(controller.URM_train)
model1.fit(alpha= 0.00022742003969239836, topK= 709, l1_ratio= 0.1488442906776265)

model2 = UserKNNCFRecommender(controller.URM_train)
model2.fit(topK= 995, shrink= 398, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

best_map = 0.0

recommender_instance = DifferentLossScoresHybridRecommender(controller.URM_train, model1, model2)

for norm in [1, 2, np.inf, -np.inf]:
    for alpha in np.arange(0.0, 1.0, 0.1):

        recommender_instance.fit(norm, alpha = alpha)

        result_df, _ = controller.evaluator_test.evaluateRecommender(recommender_instance)
        print("Norm: {}, Result: {}".format(norm, result_df.loc[10]["MAP"]))
        if result_df.loc[10]["MAP"] > best_map:
            best_map = result_df.loc[10]["MAP"]
            best_alpha = alpha
            best_norm = norm
            best_recommender = recommender_instance


cutoff = 10  # Numero di raccomandazioni da generare
recommendations_list = []
for user_id in users["user_id"]:
    recommendations = best_recommender.recommend(user_id, cutoff=cutoff)
    recommendations_list.append([user_id, recommendations])


#result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)
print (result_df.loc[10]["MAP"])
df_recommendations = pd.DataFrame(recommendations_list, columns=['user_id', 'item_list'])
df_recommendations.to_csv('recomm.csv', index=False)



