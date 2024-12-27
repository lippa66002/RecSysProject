

from EASE_R_Recommender import EASE_R_Recommender
from EASE__R_PyTorch import EASE_R_PyTorch
from ModelNames import ModelName


from ModelController import ModelController
import pandas as pd




controller = ModelController()

easer = EASE_R_PyTorch(controller.URM_train)
easer.fit(10)
cc,_ = controller.evaluator_test.evaluateRecommender(easer)
print(cc.loc[10]["MAP"])