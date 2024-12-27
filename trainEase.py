


from EASE__R_PyTorch import EASE_R_PyTorch
from ModelController import ModelController
import pandas as pd

from ModelNames import ModelName

controller = ModelController()

easer = EASE_R_PyTorch(controller.URM_train)
optuna = controller.optunizer(ModelName.EASE_R_Recommender)
cc, _ = controller.evaluator_test.evaluateRecommender(easer)
print(cc.loc[10]["MAP"])