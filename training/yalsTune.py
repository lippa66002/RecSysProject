

from ModelController import ModelController

import DataHandler
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import pandas as pd

from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

URM_all_dataframe = pd.read_csv(filepath_or_buffer="../Data/data_train.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')
users = pd.read_csv(filepath_or_buffer="../Data/data_target_users_test.csv")

ICM = pd.read_csv(filepath_or_buffer="../Data/data_ICM_metadata.csv",
                  sep=",",
                  dtype={0:int, 1:int, 2:float},
                  engine='python')
URM_all, ICM_all = DataHandler.create_urm_icm(URM_all_dataframe, ICM)
controller = ModelController(URM_all, ICM_all)

slim = SLIMElasticNetRecommender(controller.URM_all)

slim = SLIMElasticNetRecommender(controller.URM_all)
bpr = SLIM_BPR_Cython(controller.URM_train)
bpr.fit(topK =  11, learning_rate =  0.04193849345153912, lambda_i=  0.009876208709609856, lambda_j= 0.00044296738036044263, symmetric =  True, sgd_mode =  'adagrad')
slim.fit(alpha =  0.00022742003969239836, topK =  709, l1_ratio =  0.1488442906776265)
item = ItemKNNCFRecommender(controller.URM_all)
item.fit(similarity =  "cosine", topK =  8, shrink= 12)
rp3 = RP3betaRecommender(controller.URM_all)
rp3.fit(topK= 12, alpha =  0.5769111396825488, beta= 0.0019321798490027353)
hyb = ScoresHybridRecommender(controller.URM_all, slim, item, bpr, slim, slim)
hyb.fit(15,4,3,0,0)
cutoff = 10  # Numero di raccomandazioni da generare
recommendations_list = []
for user_id in users["user_id"]:
    recommendations = hyb.recommend(user_id, cutoff=cutoff)
    recommendations_list.append([user_id, recommendations])
df_recommendations = pd.DataFrame(recommendations_list, columns=['user_id', 'item_list'])
df_recommendations.to_csv('recomm.csv', index=False)