
import DataHandler
from EASE_R_Recommender import EASE_R_Recommender
from EASE__R_PyTorch import EASE_R_PyTorch
from ModelNames import ModelName


from ModelController import ModelController
import pandas as pd


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

easer = EASE_R_PyTorch(controller.URM_train)
easer.fit(10)
cc,_ = controller.evaluator_test.evaluateRecommender(easer)
print(cc.loc[10]["MAP"])