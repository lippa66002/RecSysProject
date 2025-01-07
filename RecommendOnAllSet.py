import pandas as pd

import DataHandler
from ModelController import ModelController
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import scipy.sparse as sps

from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

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

"""
recommender_instance = HybridOptunable2(controller.URM_all)

recommender_instance.load_model(folder_path="_saved_models", file_name = "HybridOptunable2 (1)")

slim_bpr = SLIM_BPR_Cython(controller.URM_all)
slim_en = SLIMElasticNetRecommender(controller.URM_all)

slim_bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender")
slim_en.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")

similarity_1 = slim_bpr.W_sparse
similarity_2 = slim_en.W_sparse

recommender_instance = ItemKNNSimilarityHybridRecommender(controller.URM_all, similarity_1, similarity_2)
recommender_instance.fit(topK= 340, alpha= 4.464670628236134e-05)


model1 = ItemKNN_CFCBF_Hybrid_Recommender(controller.URM_all, controller.ICM_all)
model2 = SLIMElasticNetRecommender(controller.URM_all)

model1.load_model(folder_path="_saved_models", file_name="ItemKNN_CFCBF_HybridRecommender")
model2.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")
recommender_instance = DifferentLossScoresHybridRecommender(controller.URM_all, model1, model2)
recommender_instance.fit(norm=1, alpha=0.0003942957276686053)

recommender_instance = SLIMElasticNetRecommender(controller.URM_all)
recommender_instance.fit(alpha= 0.0002769050189773266, topK = 979, l1_ratio= 0.15715596524094688)



model1 = RP3betaRecommender(controller.URM_train)
model2 = SLIMElasticNetRecommender(controller.URM_train)

model1.load_model(folder_path="_saved_models", file_name="RP3betaRecommender")
model2.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")
"""
stacked = sps.vstack([0.8392863849420211 * URM_all, (1 - 0.8392863849420211) * controller.ICM_all.T]).tocsr()
stacked2 = sps.vstack([0.6814451172353111 * URM_all, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()


slim_t = SLIMElasticNetRecommender(URM_all)
slim_t.load_model(folder_path="_saved_models", file_name="SLIM_ElasticNetAll")

slim_s = SLIMElasticNetRecommender(stacked2)
slim_s.load_model(folder_path="_saved_models", file_name="SLIMstackedAll1")

#bpr = SLIM_BPR_Cython(controller.URM_train)
#bpr.fit(topK =  11, learning_rate =  0.04193849345153912, lambda_i=  0.009876208709609856, lambda_j= 0.00044296738036044263, symmetric =  True, sgd_mode =  'adagrad')

#item = ItemKNNCFRecommender(controller.URM_train)
#item.fit(similarity =  "cosine", topK =  8, shrink= 12)

#rp3 = RP3betaRecommender(stacked)
#rp3.load_model(folder_path="_saved_models", file_name="rp3_stacked3_f")
#rp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)
#rp3.save_model(folder_path="_saved_models", file_name="rp3_train_f")

easeR = EASE_R_Recommender(URM_all)
easeR.load_model(folder_path="_saved_models", file_name="easeall3")
#easeR.fit(topK= 32, l2_norm= 20.402285200199643, normalize_matrix= False)
#slim.save_model(folder_path="_saved_models", file_name="ease_train_f")

p3 = P3alphaRecommender(URM_all)
#p3.fit(topK= 15, alpha= 0.5657433667229401, min_rating= 0, implicit= False, normalize_similarity= True)
p3.load_model(folder_path="_saved_models", file_name="p3alpha_all_f")

user = UserKNNCFRecommender(URM_all)
user.fit(topK= 995, shrink= 398, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

#user.load_model(folder_path="_saved_models", file_name="user_train_f")

#items = ItemKNNCBFRecommender(controller.URM_train, controller.ICM_all)
#items.fit(topK= 6, shrink= 693, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

# hyb = ItemKNN_CFCBF_Hybrid_Recommender(controller.URM_train, controller.ICM_all)
# hyb.fit(topK =  6, shrink =  167, similarity =  'asymmetric', normalize =  False, feature_weighting =  'BM25', ICM_weight =  0.375006792830105)

hyb_slims = HybridOptunable2(URM_all)
hyb_slims.fit(0.27959722573911727,slim_t,slim_s)

bestrp3 = RP3betaRecommender(URM_all)
bestrp3.load_model(folder_path="_saved_models", file_name="RP3betaAll")

hyb_best = HybridOptunable2(URM_all)
hyb_best.fit(0.18923840370620948,hyb_slims,bestrp3)

#alpha=0.9947414494756955
recommender_instance = ScoresHybridRecommender(URM_all, hyb_best, user, p3, user, p3)

# Sample x, y, and z to calculate weights
x = 0.9809789503691551
y = 0.3078230973689968

alpha = x
beta = y * (1 - x)
gamma = (1 - x) * (1 - y)

recommender_instance.fit(alpha, beta, gamma, 0, 0)


result_df, _ = controller.evaluator_test.evaluateRecommender(recommender_instance)
print(result_df)

cutoff = 10  # Numero di raccomandazioni da generare
recommendations_list = []
for user_id in users["user_id"]:
    recommendations = recommender_instance.recommend(user_id, cutoff=cutoff)
    recommendations_list.append([user_id, recommendations])


#result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)
print (result_df.loc[10]["MAP"])
df_recommendations = pd.DataFrame(recommendations_list, columns=['user_id', 'item_list'])
df_recommendations.to_csv('recomm.csv', index=False)



