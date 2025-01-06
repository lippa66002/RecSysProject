import pandas as pd

import DataHandler
from ModelController import ModelController
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import scipy.sparse as sps

from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender
from prova import recommender_instance

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
recommender_instance = HybridOptunable2(controller.URM_all)

recommender_instance.load_model(folder_path="_saved_models", file_name = "HybridOptunable2 (1)")

slim_bpr = SLIM_BPR_Cython(controller.URM_all)
slim_en = SLIMElasticNetRecommender(controller.URM_all)

slim_bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender")
slim_en.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")

similarity_1 = slim_bpr.W_sparse
similarity_2 = slim_en.W_sparse

recommender_instance = ItemKNNSimilarityHybridRecommender(controller.URM_all, similarity_1, similarity_2)
recommender_instance.fit(topK= 340, alpha= 4.464670628236134e-05)


model1 = ItemKNN_CFCBF_Hybrid_Recommender(controller.URM_all, controller.ICM_all)
model2 = SLIMElasticNetRecommender(controller.URM_all)

model1.load_model(folder_path="_saved_models", file_name="ItemKNN_CFCBF_HybridRecommender")
model2.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")
recommender_instance = DifferentLossScoresHybridRecommender(controller.URM_all, model1, model2)
recommender_instance.fit(norm=1, alpha=0.0003942957276686053)

recommender_instance = SLIMElasticNetRecommender(controller.URM_all)
recommender_instance.fit(alpha= 0.0002769050189773266, topK = 979, l1_ratio= 0.15715596524094688)



model1 = RP3betaRecommender(controller.URM_train)
model2 = SLIMElasticNetRecommender(controller.URM_train)

model1.load_model(folder_path="_saved_models", file_name="RP3betaRecommender")
model2.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")
"""
stacked = sps.vstack([0.8392863849420211 * URM_all, (1 - 0.8392863849420211) * controller.ICM_all.T]).tocsr()
slim = SLIMElasticNetRecommender(URM_all)
slim.fit(alpha= 0.00022742003969239836, topK= 709, l1_ratio= 0.1488442906776265)

#ease = EASE_R_Recommender(URM_all)
#ease.load_model(folder_path="_saved_models", file_name="easeall3")

rp3 = RP3betaRecommender(stacked)
rp3.fit(topK= 21, beta= 0.2263343041398906, alpha= 0.47403955777118195)

hyb = HybridOptunable2(URM_all)
hyb.fit(0.24002684672441646, slim, rp3)

user = UserKNNCFRecommender(URM_all)
user.fit(topK= 995, shrink= 398, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

alpha=0.9947414494756955
recommender_instance = ScoresHybridRecommender(URM_all, hyb, user, user, user, user)
recommender_instance.fit(alpha, 1-alpha, 0, 0, 0)


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



