import optuna
import scipy.sparse as sps

from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

controller = ModelController()


stacked = sps.vstack([0.8392863849420211 * controller.URM_train, (1 - 0.8392863849420211) * controller.ICM_all.T]).tocsr()
stacked2 = sps.vstack([0.6814451172353111 * controller.URM_train, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()


slim_t = SLIMElasticNetRecommender(controller.URM_train)
slim_t.load_model(folder_path="_saved_models", file_name="SLIMtrain")

slim_s = SLIMElasticNetRecommender(stacked2)
slim_s.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval1")

#bpr = SLIM_BPR_Cython(controller.URM_train)
#bpr.fit(topK =  11, learning_rate =  0.04193849345153912, lambda_i=  0.009876208709609856, lambda_j= 0.00044296738036044263, symmetric =  True, sgd_mode =  'adagrad')

#item = ItemKNNCFRecommender(controller.URM_train)
#item.fit(similarity =  "cosine", topK =  8, shrink= 12)

rp3 = RP3betaRecommender(stacked)
rp3.load_model(folder_path="_saved_models", file_name="rp3_stacked3_f")
#rp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)
#rp3.save_model(folder_path="_saved_models", file_name="rp3_train_f")

easeR = EASE_R_Recommender(controller.URM_train)
easeR.load_model(folder_path="_saved_models", file_name="easetrainll")
#easeR.fit(topK= 32, l2_norm= 20.402285200199643, normalize_matrix= False)
#slim.save_model(folder_path="_saved_models", file_name="ease_train_f")

p3 = P3alphaRecommender(controller.URM_train)
p3.load_model(folder_path="_saved_models", file_name="p3alpha_train_f")

user = UserKNNCFRecommender(controller.URM_train)
user.fit(topK= 995, shrink= 398, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

#user.load_model(folder_path="_saved_models", file_name="user_train_f")

#items = ItemKNNCBFRecommender(controller.URM_train, controller.ICM_all)
#items.fit(topK= 6, shrink= 693, similarity= 'cosine', normalize= True, feature_weighting= 'BM25')

# hyb = ItemKNN_CFCBF_Hybrid_Recommender(controller.URM_train, controller.ICM_all)
# hyb.fit(topK =  6, shrink =  167, similarity =  'asymmetric', normalize =  False, feature_weighting =  'BM25', ICM_weight =  0.375006792830105)

hyb_slims = HybridOptunable2(controller.URM_train)
hyb_slims.fit(0.27959722573911727,slim_t,slim_s)

bestrp3 = RP3betaRecommender(controller.URM_train)
bestrp3.load_model(folder_path="_saved_models", file_name="rp3train")

hyb_best = HybridOptunable2(controller.URM_train)
hyb_best.fit(0.18923840370620948,hyb_slims,bestrp3)

hyb_fin = ScoresHybridRecommender(controller.URM_train, hyb_best, user, p3, easeR, p3)

x = 0.9809789503691551
y = 0.3078230973689968

alpha = x
beta = y * (1 - x)
gamma = (1 - x) * (1 - y)

hyb_fin.fit(alpha, beta, gamma, 0, 0)

ials = IALSRecommender(controller.URM_train)
ials.load_model(folder_path="_saved_models", file_name="IALS_train")



def objective_function_scores_hybrid_6(optuna_trial):
    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    #recom1 = HybridOptunable2(controller.URM_train)
    recom = ScoresHybridRecommender(controller.URM_train, hyb_fin, ials, p3, easeR, p3)

    alpha = optuna_trial.suggest_float("alpha", 0, 1)

    recom.fit(alpha, 1-alpha, 0, 0, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_6,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)







