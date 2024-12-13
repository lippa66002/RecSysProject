import DataHandler
from ModelController import ModelController
import pandas as pd
import scipy.sparse as sps


from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
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
stacked = sps.vstack(controller.URM_train, ICM_all.T)
slim = SLIMElasticNetRecommender(stacked)
slim.fit (alpha =  0.00022742003969239836, topK =  709, l1_ratio =  0.1488442906776265)
dd, _ = controller.evaluator_test.evaluateRecommender(slim)
print(dd.loc[10]["MAP"])