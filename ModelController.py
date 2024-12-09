from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
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
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import _MatrixFactorization_Cython
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import optuna
from ModelNames import ModelName
import pandas as pd

from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender


class ModelController:

    def __init__(self, URM_all, ICM_all):
        self.URM_all = URM_all
        self.ICM_all = ICM_all
        self.URM_train, self.URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
        self.URM_train_boost, self.URM_validation = split_train_in_two_percentage_global_sample(self.URM_train, train_percentage=0.8)
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
            model = MultVAERecommender_PyTorch(self.URM_train)
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
            model1 = SLIM_BPR_Cython(self.URM_train)
            model2 = SLIMElasticNetRecommender(self.URM_train)

            model1.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender")
            model2.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")
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
        else:
            raise ValueError("Model not found")

        return model


    def optunizer(self, model_name):

        if model_name == ModelName.SLIM_ElasticNet:
            obj_func = self.objective_function_SLIM
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
        elif model_name == ModelName.ContentBasedRecommender:
            obj_func = self.objective_function_content_based
        elif model_name == ModelName.MatrixFactorization_Cython_Recommender:
            obj_func = self.objective_function_matrixFactorizationCython
        else:
            raise ValueError("Model not found")

        optuna_study = optuna.create_study(direction="maximize")

        save_results = SaveResults()

        optuna_study.optimize(obj_func,
                              callbacks=[save_results],
                              n_trials=50)

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

        recommender_instance = SLIMElasticNetRecommender(self.URM_train)
        full_hyperp = {"alpha": optuna_trial.suggest_float("alpha", 0.0001, 0.0003, log=True), #1e-5, 1e-3
                       "topK": optuna_trial.suggest_int("topK", 600, 1000), #5, 1000
                       "l1_ratio": optuna_trial.suggest_float("l1_ratio", 0.1, 0.4), #1e-3, 0.6
                       }
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
        recommender_instance = EASE_R_Recommender(self.URM_train)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "l2_norm": optuna_trial.suggest_float("l2_norm", 1e-3, 1e3, log=True),
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

    def objective_function_multVAE(self, optuna_trial):
        recommender_instance = MultVAERecommender_PyTorch(self.URM_train, use_gpu=True)
        full_hyperp = {
            "epochs": optuna_trial.suggest_int("epochs", 10, 100),
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
            #"batch_size": optuna_trial.suggest_int("batch_size", 32, 512),
            "symmetric": optuna_trial.suggest_categorical("symmetric", [True, False]),
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
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
