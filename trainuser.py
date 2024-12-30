from ModelController import ModelController
from ModelNames import ModelName

print("USER")
controller = ModelController()
optuna = controller.optunizer(ModelName.UserKNNCFRecommender)