import DataHandler
from EASE_R_Recommender import EASE_R_Recommender
from EASE__R_PyTorch import EASE_R_PyTorch
from ModelNames import ModelName


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


controller = ModelController()
ease1 = EASE_R_Recommender(controller.URM_train)
ease1.fit(topK=201,l2_norm=22.48985827601,normalize_matrix="false")
ease1.save_model(folder_path="_saved_models", file_name="easetrain")
ease2 = EASE_R_Recommender(URM_all)
ease2.fit(topK=201,l2_norm=22.48985827601,normalize_matrix="false")
ease2.save_model(folder_path="_saved_models", file_name="easeall")
optuna = controller.optunizer(ModelName.EASE_R_Recommender)