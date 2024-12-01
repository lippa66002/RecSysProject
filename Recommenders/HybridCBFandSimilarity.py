from sympy.codegen.ast import continue_


class HybridRecommenderCBFandSimilarity(BaseItemSimilarityMatrixRecommender):
    def __init__(self, URM_train, ICM_train=None, slim_recommender=None, cbf_recommender=None, alpha=0.5,
                 verbose=True):
        super(HybridRecommender, self).__init__(URM_train, verbose=verbose)
        self.alpha = alpha

        self.slim_recommender = SLIMElasticNetRecommender(URM_train, verbose=verbose)
        self.cbf_recommender = ItemKNNCBFRecommender(URM_train, ICM_train, verbose=verbose)

        self.slim_recommender = self.slim_recommender.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")
        self.cbf_recommender = self.cbf_recommender.load_model(folder_path="_saved_models", file_name="ItemKNNCBFRecommender")

    def fit(self, slim_params=None, cbf_params=None):
        if(slim_params is not None):
            self.slim_recommender.fit(**slim_params)
        if(cbf_params is not None):
            self.cbf_recommender.fit(**cbf_params)

    def compute_score(self, user_id):
        # Usa i punteggi pre-calcolati dai modelli
        slim_scores = self.slim_recommender.compute_score(user_id)
        cbf_scores = self.cbf_recommender.compute_score(user_id)
        return self.alpha * slim_scores + (1 - self.alpha) * cbf_scores

    def recommend(self, user_id, top_n=10):
        scores = self.compute_score(user_id)
        return np.argsort(-scores)[:top_n]
