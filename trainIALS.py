import pandas as pd

import DataHandler
from ModelController import ModelController
from ModelNames import ModelName
import scipy.sparse as sps

from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

URM_all_dataframe = pd.read_csv(filepath_or_buffer="Data/data_train.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')
users = pd.read_csv(filepath_or_buffer="Data/data_target_users_test.csv")

ICM = pd.read_csv(filepath_or_buffer="Data/data_ICM_metadata.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')

controller = ModelController()
URM_all, ICM_all = DataHandler.create_urm_icm(URM_all_dataframe, ICM)

ials = IALSRecommender(controller.URM_boost)
ials.fit(confidence_scaling= 'linear', num_factors=173, alpha= 6.7979092186218715, epsilon= 1.3000803763047268, reg=0.03413944416805373, init_std= 0.06581036009302164)
ials.save_model(folder_path="_saved_models", file_name="IALSboost")