from ModelController import ModelController
from ModelNames import ModelName
print("ITEMCF")
controller = ModelController()
optuna = controller.optunizer(ModelName.ItemKNNCFRecommender)