import DataHandler
from EASE_R_Recommender import EASE_R_Recommender
from ModelNames import ModelName
from ModelController import ModelController
import pandas as pd
controller = ModelController()

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


optuna = controller.optunizer(ModelName.EASE_R_Recommender)