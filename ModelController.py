import zipfile

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from EASE__R_PyTorch import EASE_R_PyTorch
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMItemHybridRecommender
from Recommenders.FeatureWeighting.Cython.CFW_DVV_Similarity_Cython import CFW_DVV_Similarity_Cython
from Recommenders.FeatureWeighting.Cython.CFW_D_Similarity_Cython import CFW_D_Similarity_Cython
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Optimize.SaveResults import SaveResults
from Recommenders.HybridDifferentLossFunctions import DifferentLossScoresHybridRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import _MatrixFactorization_Cython
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import optuna
from ModelNames import ModelName
import scipy.sparse as sps
import pandas as pd
import os
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender
from easeR_Filo import EASE_R_Filo


class ModelController:

    def __init__(self):
        # Get the directory of the current script
        current_dir = os.path.dirname(__file__)
        # Construct the path to the zip file
        zip_file_path = os.path.join(current_dir, 'matrici_sparse.zip')
        with zipfile.ZipFile(zip_file_path, "r") as zipf:
            zipf.extract("URM_trainval.npz")
            zipf.extract("URM_test.npz")
            zipf.extract("URM_train.npz")
            zipf.extract("URM_validation.npz")
            zipf.extract("ICM_all.npz")
            zipf.close()

        # Carica la matrice sparse
        self.URM_train = sps.load_npz("URM_trainval.npz")
        self.URM_boost = sps.load_npz("URM_train.npz")
        self.URM_test = sps.load_npz("URM_test.npz")
        self.URM_validation = sps.load_npz("URM_validation.npz")
        self.ICM_all = sps.load_npz("ICM_all.npz")
        self.evaluator_validation = EvaluatorHoldout(self.URM_validation, cutoff_list=[10])
        self.evaluator_test = EvaluatorHoldout(self.URM_test, cutoff_list=[10])

    def generate_model(self, model_name, optuna_hpp):

        if model_name == ModelName.SLIM_ElasticNet:
            model = SLIMElasticNetRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.ItemKNNCFRecommender:
            model = ItemKNNCFRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.RP3betaRecommender:
            model = RP3betaRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.ContentBasedRecommender:
            model = ItemKNNCBFRecommender(self.URM_train, self.ICM_all)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.EASE_R_Recommender:
            model = EASE_R_Recommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.HybridOptunable2:
            model1 = SLIM_BPR_Cython(self.URM_train)
            model1.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender")
            model2 = ItemKNNCFRecommender(self.URM_train)
            model2.load_model(folder_path="_saved_models", file_name="ItemKNNCFRecommender")
            model = HybridOptunable2(self.URM_train)
            model.fit(optuna_hpp.pop("alpha"), model1, model2)

            #model = HybridOptunable2(self.URM_train)
            #model.fit(**optuna_hpp)
        elif model_name == ModelName.MultVAERecommender_PyTorch:
            model = MultVAERecommender_PyTorch(self.URM_train, use_gpu=True)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.MultVAERecommender:
            model = MultVAERecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.IALSRecommender:
            model = IALSRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.UserKNNCFRecommender:
            model = UserKNNCFRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.Hybrid_ItemKNN_CF_CBF:
            model = ItemKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM_all)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.ItemKNNSimilarityHybridRecommender:
            slim_bpr = SLIM_BPR_Cython(self.URM_train)
            slim_en = SLIMElasticNetRecommender(self.URM_train)

            slim_bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender")
            slim_en.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")

            similarity_1 = slim_bpr.W_sparse
            similarity_2 = slim_en.W_sparse
            model = ItemKNNSimilarityHybridRecommender(self.URM_train, similarity_1, similarity_2)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.SLIM_BPR_Recommender:
            model = SLIM_BPR_Cython(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.P3alphaRecommender:
            model = P3alphaRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.PureSVDRecommender:
            model = PureSVDRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.ScoresHybridRecommender:
            #model1 = SLIM_BPR_Cython(self.URM_train)
            model1 = ItemKNNCFRecommender(self.URM_train)
            model2 = SLIMElasticNetRecommender(self.URM_train)

            #model1.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender")
            model1.fit(similarity= "cosine", topK= 8, shrink= 12)
            model2.load_model(folder_path="_saved_models", file_name="SLIM_ElasticNetTrain")
            model = ScoresHybridRecommender(self.URM_train, model1, model2)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.DifferentLossScoresHybridRecommender:
            model1 = RP3betaRecommender(self.URM_train)
            model2 = SLIMElasticNetRecommender(self.URM_train)

            model1.load_model(folder_path="_saved_models", file_name="RP3betaRecommender")
            model2.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")
            model = DifferentLossScoresHybridRecommender(self.URM_train, model1, model2)
            model.fit(**optuna_hpp)
        #elif model_name == ModelName.CFW_D_Similarity_Cython:
         #   model = CFW_D_Similarity_Cython(self.URM_train)
         #   model.fit(**optuna_hpp)
        elif model_name == ModelName.MatrixFactorization_Cython_Recommender:
            model = _MatrixFactorization_Cython(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.ItemKNNCBFRecommender:
            model = ItemKNNCBFRecommender(self.URM_train, self.ICM_all)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.CFW_DVV_Similarity_Cython:
            model = SLIMElasticNetRecommender(self.URM_train)
            model.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")
            model = CFW_DVV_Similarity_Cython(self.URM_train, self.ICM_all, model.W_sparse)
            model.fit(**optuna_hpp)
        elif model_name  == ModelName.LightFMItemHybridRecommender:
            model = LightFMItemHybridRecommender(self.URM_train, self.ICM_all)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.Hybrid_UserKNN_CF_CBF:
            model = UserKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM_all)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.EASE_R_Filo:
            model = EASE_R_Filo(self.URM_train)
            model.fit(**optuna_hpp)
        else:
            raise ValueError("Model not found")

        return model


    def optunizer(self, model_name):


        if model_name == ModelName.SLIM_ElasticNet:
            obj_func = self.objective_function_SLIM
        elif model_name == ModelName.IALSRecommender:
            obj_func = self.objective_function_IALS
        elif model_name == ModelName.ItemKNNCFRecommender:
            obj_func = self.objective_function_KNN_similarities
        elif model_name == ModelName.RP3betaRecommender:
            obj_func = self.objective_function_graph
        elif model_name == ModelName.EASE_R_Recommender:
            obj_func = self.objective_function_easeR
        elif model_name == ModelName.HybridOptunable2:
            #obj_func = self.objective_function_hybridOptunable2
            obj_func = self.obj_hybrid
        elif model_name == ModelName.MultVAERecommender_PyTorch:
            obj_func = self.objective_function_multVAE_pytorch
        elif model_name == ModelName.MultVAERecommender:
            obj_func = self.objective_function_multVAE
        elif model_name == ModelName.UserKNNCFRecommender:
            obj_func = self.objective_function_userKNN
        elif model_name == ModelName.Hybrid_ItemKNN_CF_CBF:
            obj_func = self.objective_function_hybrid_ItemKNN_CF_CBF
        elif model_name == ModelName.ItemKNNSimilarityHybridRecommender:
            obj_func = self.objective_function_itemKNN_similarity_hybrid
        elif model_name == ModelName.SLIM_BPR_Recommender:
            obj_func = self.objective_function_SLIM_BPR_Cython
        elif model_name == ModelName.P3alphaRecommender:
            obj_func = self.objective_function_P3alpha
        elif model_name == ModelName.PureSVDRecommender:
            obj_func = self.objective_function_PureSVD
        elif model_name == ModelName.DifferentLossScoresHybridRecommender:
            obj_func = self.objective_function_hybrid_different_loss_scores
        elif model_name == ModelName.ScoresHybridRecommender:
            obj_func = self.objective_function_scores_hybrid
        elif model_name == ModelName.ContentBasedRecommender:
            obj_func = self.objective_function_content_based
        elif model_name == ModelName.MatrixFactorization_Cython_Recommender:
            obj_func = self.objective_function_matrixFactorizationCython
        elif model_name == ModelName.ItemKNNCBFRecommender:
            obj_func = self.objective_function_itemKNNCBF
        elif model_name == ModelName.CFW_DVV_Similarity_Cython:
            obj_func = self.objective_function_CFW_DVV_Similarity_Cython
        elif model_name == ModelName.LightFMItemHybridRecommender:
            obj_func = self.objective_function_LightFMItemHybridRecommender
        elif model_name == ModelName.Hybrid_UserKNN_CF_CBF:
            obj_func = self.objective_function_hybrid_UserKNN_CF_CBF
        elif model_name == ModelName.EASE_R_Filo:
            obj_func = self.objective_function_easeR_filo
        else:
            raise ValueError("Model not found")

        optuna_study = optuna.create_study(direction="maximize")

        save_results = SaveResults()

        optuna_study.optimize(obj_func,
                              callbacks=[save_results],
                              n_trials=500)

        print(save_results.results_df)
        print(optuna_study.best_trial.params)

        return optuna_study.best_trial.params

    def objective_function_KNN_similarities(self, optuna_trial):

        recommender_instance = ItemKNNCFRecommender(self.URM_train)
        similarity = optuna_trial.suggest_categorical("similarity",
                                                      ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky',
                                                       'euclidean'])

        full_hyperp = {"similarity": similarity,
                       "topK": optuna_trial.suggest_int("topK", 5, 1000),
                       "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
                       }

        if similarity == "asymmetric":
            full_hyperp["asymmetric_alpha"] = optuna_trial.suggest_float("asymmetric_alpha", 0, 2, log=False)
            full_hyperp["normalize"] = True

        elif similarity == "tversky":
            full_hyperp["tversky_alpha"] = optuna_trial.suggest_float("tversky_alpha", 0, 2, log=False)
            full_hyperp["tversky_beta"] = optuna_trial.suggest_float("tversky_beta", 0, 2, log=False)
            full_hyperp["normalize"] = True

        elif similarity == "euclidean":
            full_hyperp["normalize_avg_row"] = optuna_trial.suggest_categorical("normalize_avg_row", [True, False])
            full_hyperp["similarity_from_distance_mode"] = optuna_trial.suggest_categorical(
                "similarity_from_distance_mode", ["lin", "log", "exp"])
            full_hyperp["normalize"] = optuna_trial.suggest_categorical("normalize", [True, False])

        recommender_instance.fit(**full_hyperp)

        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)

        return result_df.loc[10]["MAP"]

    def objective_function_SLIM(self, optuna_trial):



        # Creazione dell'istanza del recommender
        recommender_instance = SLIMElasticNetRecommender(self.URM_train)

        # Hyperparametri da ottimizzare
        full_hyperp = {
            "alpha": optuna_trial.suggest_float("alpha", 0.00007, 0.0003, log=True),  # 1e-5, 1e-3 fino a 0.0003
            "topK": optuna_trial.suggest_int("topK", 600, 1000),  # 5, 1000
            "l1_ratio": optuna_trial.suggest_float("l1_ratio", 0.01, 0.4),  # 1e-3, 0.6 fino a 0.4
        }

        # Allenamento del recommender
        recommender_instance.fit(**full_hyperp)

        # Valutazione del recommender
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)

        # Restituzione del MAP
        return result_df.loc[10]["MAP"]

    def objective_function_IALS(self,optuna_trial):
        recommender_instance = IALSRecommender(self.URM_train)
        confidence_scaling = optuna_trial.suggest_categorical("confidence_scaling",
                                                      ['linear', 'log'])
        full_hyperp = {"num_factors": optuna_trial.suggest_int("num_factors", 10, 200),
            "confidence_scaling" : confidence_scaling,
            "alpha": optuna_trial.suggest_float("alpha", 0.1, 50.0),
            "epsilon": optuna_trial.suggest_float("epsilon", 0.1, 10.0),
            "reg": optuna_trial.suggest_float("reg", 1e-3, 1e-1),

            "init_std" : optuna_trial.suggest_float("init_std", 0.01, 0.2)}
        recommender_instance.fit(**full_hyperp)

        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)

        return result_df.loc[10]["MAP"]

    def objective_function_graph(self, optuna_trial):
        recomm = RP3betaRecommender(self.URM_train)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "beta": optuna_trial.suggest_float("beta", 0, 1),
            "alpha": optuna_trial.suggest_float("alpha", 0, 1.5)}
        recomm.fit(**full_hyperp)

        result_df, _ = self.evaluator_test.evaluateRecommender(recomm)

        return result_df.loc[10]["MAP"]

    def objective_function_easeR(self, optuna_trial):
        recommender_instance = EASE_R_PyTorch(self.URM_train)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "l2": optuna_trial.suggest_float("l2", 1e-3, 1e3, log=True),
            "normalize_matrix": optuna_trial.suggest_categorical("normalize_matrix", [True, False])
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_hybridOptunable2(self, optuna_trial, itemKnncf, slimrec):
        print("helloworld")
        alpha = optuna_trial.suggest_float("alpha", 0.1, 0.9)
        recommender_object = HybridOptunable2(self.URM_train)
        recommender_object.fit(alpha, itemKnncf, slimrec)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_object)
        return result_df.loc[10]["MAP"]

    def objective_function_multVAE_pytorch(self, optuna_trial):
        recommender_instance = MultVAERecommender_PyTorch(self.URM_train, use_gpu=True)
        full_hyperp = {
            #"epochs": optuna_trial.suggest_int("epochs", 10, 100),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": optuna_trial.suggest_int("batch_size", 32, 512),
            "dropout": optuna_trial.suggest_float("dropout", 0.1, 0.5),
            "total_anneal_steps": optuna_trial.suggest_int("total_anneal_steps", 10000, 200000),
            "anneal_cap": optuna_trial.suggest_float("anneal_cap", 0.1, 0.5),
            "p_dims": [self.URM_train.shape[1], self.URM_train.shape[1]]
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_multVAE(self, optuna_trial):
        recommender_instance = MultVAERecommender(self.URM_train)
        full_hyperp = {
            #"epochs": optuna_trial.suggest_int("epochs", 10, 100),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": optuna_trial.suggest_int("batch_size", 16, 128),
            "dropout": optuna_trial.suggest_float("dropout", 0.1, 0.5),
            "total_anneal_steps": optuna_trial.suggest_int("total_anneal_steps", 10000, 200000),
            "anneal_cap": optuna_trial.suggest_float("anneal_cap", 0.1, 0.5),
            "p_dims": [self.URM_train.shape[1] // 2, self.URM_train.shape[1] // 4] #"p_dims": [self.URM_train.shape[1], self.URM_train.shape[1]]

        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_userKNN(self, optuna_trial):
        recommender_instance = UserKNNCFRecommender(self.URM_train)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
            "similarity": optuna_trial.suggest_categorical("similarity",
                                                           ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky',
                                                            'euclidean']),
            "normalize": optuna_trial.suggest_categorical("normalize", [True, False]),
            "feature_weighting": optuna_trial.suggest_categorical("feature_weighting", ["BM25", "TF-IDF", "none"])
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_hybrid_ItemKNN_CF_CBF(self, optuna_trial):
        recommender_instance = ItemKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM_all)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
            "similarity": optuna_trial.suggest_categorical("similarity",
                                                           ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky',
                                                            'euclidean']),
            "normalize": optuna_trial.suggest_categorical("normalize", [True, False]),
            "feature_weighting": optuna_trial.suggest_categorical("feature_weighting", ["BM25", "TF-IDF", "none"]),
            "ICM_weight": optuna_trial.suggest_float("ICM_weight", 0.1, 1.0)
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_itemKNN_similarity_hybrid(self, optuna_trial):
        slim_bpr = SLIM_BPR_Cython(self.URM_train)
        slim_en = SLIMElasticNetRecommender(self.URM_train)

        slim_bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender")
        slim_en.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")

        similarity_1 = slim_bpr.W_sparse
        similarity_2 = slim_en.W_sparse

        recommender_instance = ItemKNNSimilarityHybridRecommender(self.URM_train, similarity_1, similarity_2)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "alpha": optuna_trial.suggest_float("alpha", 0.0, 1.0)
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_SLIM_BPR_Cython(self, optuna_trial):
        recommender_instance = SLIM_BPR_Cython(self.URM_train)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "lambda_i": optuna_trial.suggest_float("lambda_i", 1e-4, 1e-2, log=True),
            "lambda_j": optuna_trial.suggest_float("lambda_j", 1e-4, 1e-2, log=True),
            "symmetric": optuna_trial.suggest_categorical("symmetric", [True, False]),
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"])
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_P3alpha(self, optuna_trial):
        recommender_instance = P3alphaRecommender(self.URM_train)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "alpha": optuna_trial.suggest_float("alpha", 0.0, 2.0),
            "min_rating": optuna_trial.suggest_int("min_rating", 0, 10),
            "implicit": optuna_trial.suggest_categorical("implicit", [True, False]),
            "normalize_similarity": optuna_trial.suggest_categorical("normalize_similarity", [True, False])
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_PureSVD(self, optuna_trial):
        recommender_instance = PureSVDRecommender(self.URM_train)
        full_hyperp = {
            "num_factors": optuna_trial.suggest_int("num_factors", 10, 500),
            "random_seed": optuna_trial.suggest_int("random_seed", 0, 10000)
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def obj_hybrid(self, optuna_trial):
        model1 = SLIM_BPR_Cython(self.URM_train)
        model1.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender")
        model2 = ItemKNNCFRecommender(self.URM_train)
        model2.load_model(folder_path="_saved_models", file_name="ItemKNNCFRecommender")
        alpha = optuna_trial.suggest_float("alpha", 0.1, 0.9)
        recommender_object = HybridOptunable2(self.URM_train)
        recommender_object.fit(alpha, model1, model2)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_object)
        return result_df.loc[10]["MAP"]

    def objective_function_content_based(self, optuna_trial):
        recommender_instance = ItemKNNCBFRecommender(self.URM_train, self.ICM_all)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
            "similarity": optuna_trial.suggest_categorical("similarity",
                                                           ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky',
                                                            'euclidean']),
            "normalize": optuna_trial.suggest_categorical("normalize", [True, False]),
            "feature_weighting": optuna_trial.suggest_categorical("feature_weighting", ["BM25", "TF-IDF", "none"])
        }

        if full_hyperp["similarity"] == "asymmetric":
            full_hyperp["asymmetric_alpha"] = optuna_trial.suggest_float("asymmetric_alpha", 0, 2, log=False)

        elif full_hyperp["similarity"] == "tversky":
            full_hyperp["tversky_alpha"] = optuna_trial.suggest_float("tversky_alpha", 0, 2, log=False)
            full_hyperp["tversky_beta"] = optuna_trial.suggest_float("tversky_beta", 0, 2, log=False)

        elif full_hyperp["similarity"] == "euclidean":
            full_hyperp["normalize_avg_row"] = optuna_trial.suggest_categorical("normalize_avg_row", [True, False])
            full_hyperp["similarity_from_distance_mode"] = optuna_trial.suggest_categorical(
                "similarity_from_distance_mode", ["lin", "log", "exp"])

        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_hybrid_different_loss_scores(self, optuna_trial):
        model1 = RP3betaRecommender(self.URM_train)
        model1.load_model(folder_path="_saved_models", file_name="RP3betaRecommender")
        model2 = SLIMElasticNetRecommender(self.URM_train)
        model2.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")

        recommender_instance = DifferentLossScoresHybridRecommender(self.URM_train, model1, model2)
        full_hyperp = {
            "norm": optuna_trial.suggest_categorical("norm", [1, 2]),
            "alpha": optuna_trial.suggest_float("alpha", 0.0, 1.0)
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_itemKNNCBF(self, optuna_trial):
        recommender_instance = ItemKNNCBFRecommender(self.URM_train, self.ICM_all)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
            "similarity": optuna_trial.suggest_categorical("similarity",
                                                           ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky',
                                                            'euclidean']),
            "normalize": optuna_trial.suggest_categorical("normalize", [True, False]),
            "feature_weighting": optuna_trial.suggest_categorical("feature_weighting", ["BM25", "TF-IDF", "none"])
        }

        if full_hyperp["similarity"] == "asymmetric":
            full_hyperp["asymmetric_alpha"] = optuna_trial.suggest_float("asymmetric_alpha", 0, 2, log=False)

        elif full_hyperp["similarity"] == "tversky":
            full_hyperp["tversky_alpha"] = optuna_trial.suggest_float("tversky_alpha", 0, 2, log=False)
            full_hyperp["tversky_beta"] = optuna_trial.suggest_float("tversky_beta", 0, 2, log=False)

        elif full_hyperp["similarity"] == "euclidean":
            full_hyperp["normalize_avg_row"] = optuna_trial.suggest_categorical("normalize_avg_row", [True, False])
            full_hyperp["similarity_from_distance_mode"] = optuna_trial.suggest_categorical(
                "similarity_from_distance_mode", ["lin", "log", "exp"])

        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_CFW_DVV_Similarity_Cython(self, optuna_trial):
        model = SLIMElasticNetRecommender(self.URM_train)
        model.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")
        recommender_instance = CFW_DVV_Similarity_Cython(self.URM_train, self.ICM_all, model.W_sparse)

        full_hyperp = {
            "n_factors": optuna_trial.suggest_int("n_factors", 10, 100),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "l2_reg_D": optuna_trial.suggest_float("l2_reg_D", 1e-5, 1e-2, log=True),
            "l2_reg_V": optuna_trial.suggest_float("l2_reg_V", 1e-5, 1e-2, log=True),
            "epochs": optuna_trial.suggest_int("epochs", 10, 50),
            "topK": optuna_trial.suggest_int("topK", 50, 500),
            "positive_only_weights": optuna_trial.suggest_categorical("positive_only_weights", [True, False]),
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["adam", "adagrad"]),
        }

        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_scores_hybrid(self, optuna_trial):
        slim = SLIMElasticNetRecommender(self.URM_train)
        bpr = SLIM_BPR_Cython(self.URM_train)
        bpr.fit(topK =  11, learning_rate =  0.04193849345153912, lambda_i=  0.009876208709609856, lambda_j= 0.00044296738036044263, symmetric =  True, sgd_mode =  'adagrad')
        slim.fit(alpha =  0.00022742003969239836, topK =  709, l1_ratio =  0.1488442906776265)
        item = ItemKNNCFRecommender(self.URM_train)
        item.fit(similarity =  "cosine", topK =  8, shrink= 12)
        rp3 = RP3betaRecommender(self.URM_train)
        rp3.fit(topK= 12, alpha =  0.5769111396825488, beta= 0.0019321798490027353)
        user = UserKNNCFRecommender(self.URM_train)
        user.fit(similarity =  "dice", topK= 19, shrink= 737)
        items = ItemKNNCBFRecommender(self.URM_train, self.ICM_all)
        items.load_model.fit()
        hyb = ItemKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM_all)
        hyb.fit(topK =  6, shrink =  167, similarity =  'asymmetric', normalize =  False, feature_weighting =  'BM25', ICM_weight =  0.375006792830105)
        #bpr = SLIM_BPR_Cython(self.URM_train)
        #bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
        recom1 = ScoresHybridRecommender(self.URM_train, slim, item,rp3,bpr,items)
        recom2 = ScoresHybridRecommender(self.URM_train, slim, bpr,rp3,bpr,items)
        recom3 = ScoresHybridRecommender(self.URM_train, slim, items,rp3,bpr,items)
        recom4 = ScoresHybridRecommender(self.URM_train, slim, hyb,rp3,bpr,items)
        recom5 = ScoresHybridRecommender(self.URM_train, slim, rp3,rp3,bpr,items)
        recom6 = ScoresHybridRecommender(self.URM_train, slim, user,rp3,bpr,items)

        '''full_hyperp = {

            "alpha": optuna_trial.suggest_float("alpha", 0.0, 1.0),
            "beta": optuna_trial.suggest_float("beta", 0.0, 1.0),
            "gamma": optuna_trial.suggest_float("gamma", 0.0, 1.0),
            "delta": optuna_trial.suggest_float("delta", 0.0, 1.0),
            "epsilon": optuna_trial.suggest_float("epsilon", 0.0, 1.0)
        }'''
        alpha = optuna_trial.suggest_float("alpha", 0.0, 1.0)
        beta= optuna_trial.suggest_float("beta", 0.0, 1.0)
        recom1.fit(alpha,beta,0,0,0)
        recom2.fit(alpha, beta, 0, 0, 0)
        recom3.fit(alpha, beta, 0, 0, 0)
        recom4.fit(alpha, beta, 0, 0, 0)
        recom5.fit(alpha, beta, 0, 0, 0)
        recom6.fit(alpha, beta, 0, 0, 0)
        result_df, _ = self.evaluator_test.evaluateRecommender(recom1)
        return result_df.loc[10]["MAP"]

    def objective_function_matrixFactorizationCython(self, optuna_trial):
        recommender_instance = _MatrixFactorization_Cython(self.URM_train)

        full_hyperp = {
            "epochs": optuna_trial.suggest_int("epochs", 100, 100),
            "batch_size": optuna_trial.suggest_int("batch_size", 1, 1024),
            "num_factors": optuna_trial.suggest_int("num_factors", 1, 200),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adam", "adagrad", "rmsprop"])
        }

        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_LightFMItemHybridRecommender(self, optuna_trial):
        recommender_instance = LightFMItemHybridRecommender(self.URM_train, self.ICM_all)

        full_hyperp = {
            "epochs": optuna_trial.suggest_int("epochs", 10, 100),
            "loss": optuna_trial.suggest_categorical("loss", ["bpr", "warp", "warp-kos"]),
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["adagrad", "adadelta"]),
            "n_components": optuna_trial.suggest_int("n_components", 10, 200),
            "item_alpha": optuna_trial.suggest_float("item_alpha", 1e-6, 1e-1, log=True),
            "user_alpha": optuna_trial.suggest_float("user_alpha", 1e-6, 1e-1, log=True),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        }

        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)

        return result_df.loc[10]["MAP"]

    def objective_function_hybrid_UserKNN_CF_CBF(self, optuna_trial):

        recommender_instance = UserKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM_all)

        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
            "similarity": optuna_trial.suggest_categorical("similarity",
                                                           ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky',
                                                            'euclidean']),
            "normalize": optuna_trial.suggest_categorical("normalize", [True, False]),
            "feature_weighting": optuna_trial.suggest_categorical("feature_weighting", ["BM25", "TF-IDF", "none"]),
            "UCM_weight": optuna_trial.suggest_float("UCM_weight", 0.1, 1.0)
        }

        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)

        return result_df.loc[10]["MAP"]

    def objective_function_easeR_filo(self, optuna_trial):
        recommender_instance = EASE_R_Filo(self.URM_train)
        full_hyperp = {
            "topK" : optuna_trial.suggest_int("topK", 10, 500),
            "l2_norm" : optuna_trial.suggest_float("l2_norm", 1e2, 1e5, log=True),
            "normalize_matrix" : optuna_trial.suggest_categorical("normalize_matrix", [False, True])
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]
