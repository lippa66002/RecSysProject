


from EASE__R_PyTorch import EASE_R_PyTorch
from ModelController import ModelController
import pandas as pd

from ModelNames import ModelName

controller = ModelController()


optuna = controller.optunizer(ModelName.EASE_R_Recommender)
