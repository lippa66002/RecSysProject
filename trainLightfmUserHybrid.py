from ModelController import ModelController
from ModelNames import ModelName

controller = ModelController()

optuna = controller.optunizer(ModelName.LightFMUserHybridRecommender)