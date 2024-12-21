
import pandas as pd
import DataHandler
from ModelController import ModelController
from ModelNames import ModelName


from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
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
slim1 = SLIMElasticNetRecommender(controller.URM_train)
slim1.fit(topK=709, l1_ratio= 0.1488442906776265, alpha=0.00018954228953640845)
r, _ = controller.evaluator_test.evaluateRecommender(slim1)
print(r.loc[10]["MAP"])
slim1 = SLIMElasticNetRecommender(controller.URM_train)
slim1.fit(topK=653, l1_ratio= 0.1488442906776265, alpha=0.00018954228953640845)
r, _ = controller.evaluator_test.evaluateRecommender(slim1)
print(r.loc[10]["MAP"])
slim1 = SLIMElasticNetRecommender(controller.URM_train)
slim1.fit(topK=849, l1_ratio= 0.09126171988570234, alpha=0.00024173941597362773)
r, _ = controller.evaluator_test.evaluateRecommender(slim1)
print(r.loc[10]["MAP"])
slim1 = SLIMElasticNetRecommender(controller.URM_train)
slim1.fit(topK=742, l1_ratio= 0.12812423236902357, alpha=0.00025558622938992706)
r, _ = controller.evaluator_test.evaluateRecommender(slim1)
print(r.loc[10]["MAP"])
slim1 = SLIMElasticNetRecommender(controller.URM_train)
slim1.fit(topK=957, l1_ratio= 0.30508795280552997, alpha=0.00011515224032129526)
r, _ = controller.evaluator_test.evaluateRecommender(slim1)
print(r.loc[10]["MAP"])
slim1 = SLIMElasticNetRecommender(controller.URM_train)
slim1.fit(topK=613, l1_ratio= 0.2025689481426673, alpha=0.00021586771546520844)
r, _ = controller.evaluator_test.evaluateRecommender(slim1)
print(r.loc[10]["MAP"])

