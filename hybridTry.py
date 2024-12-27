import DataHandler
from ModelController import ModelController
from ModelNames import ModelName
import numpy as np
import pandas as pd
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

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

model1 = ItemKNNCFRecommender(controller.URM_train)
model1.fit(similarity= "cosine", topK= 8, shrink= 12)

model2 = RP3betaRecommender(controller.URM_train)
model2.fit(topK= 12, alpha= 0.5769111396825488, beta= 0.0019321798490027353)

best_map = 0.0

for alpha in np.arange(0.7, 0.8, 0.01):
    new_similarity = (1 - alpha) * model1.W_sparse + alpha * model2.W_sparse
    recommender_instance = ItemKNNCustomSimilarityRecommender(controller.URM_train)
    recommender_instance.fit(new_similarity)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recommender_instance)
    print("Alpha: {}, Result: {}".format(alpha, result_df.loc[10]["MAP"]))

    if result_df.loc[10]["MAP"] > best_map:
        best_map = result_df.loc[10]["MAP"]
        best_alpha = alpha
        best_recommender = recommender_instance

cutoff = 10  # Numero di raccomandazioni da generare
recommendations_list = []
for user_id in users["user_id"]:
    recommendations = best_recommender.recommend(user_id, cutoff=cutoff)
    recommendations_list.append([user_id, recommendations])


#result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)
print (result_df.loc[10]["MAP"])
df_recommendations = pd.DataFrame(recommendations_list, columns=['user_id', 'item_list'])
df_recommendations.to_csv('recomm.csv', index=False)



