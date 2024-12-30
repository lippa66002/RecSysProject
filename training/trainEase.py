

from EASE_R_Recommender import EASE_R_Recommender
from EASE__R_PyTorch import EASE_R_PyTorch
from ModelNames import ModelName


from ModelController import ModelController
import pandas as pd




controller = ModelController()

optuna = controller.optunizer(ModelName.EASE_R_Recommender)