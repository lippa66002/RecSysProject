
import DataHandler
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

from ModelController import ModelController
import pandas as pd


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

ease = EASE_R_Recommender(controller.URM_train)
ease.fit()
dd, _ = controller.evaluator_test.evaluateRecommender(ease)
print(dd.loc[10]["MAP"])