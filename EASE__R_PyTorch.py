import torch
import numpy as np
import time
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import similarityMatrixTopK, check_matrix
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity

class EASE_R_PyTorch(BaseItemSimilarityMatrixRecommender):
    """ EASE_R_PyTorch - Versione con PyTorch"""

    RECOMMENDER_NAME = "EASE_R_PyTorch"

    def __init__(self, URM_train, sparse_threshold_quota=None):
        super(EASE_R_PyTorch, self).__init__(URM_train)
        self.sparse_threshold_quota = sparse_threshold_quota

    def fit(self, topK=None, normalize_matrix=False, verbose=True):
        """
        topK: number of top-K items to use for similarity computation
        normalize_matrix: whether to normalize the matrix or not
        verbose: whether to print progress
        """
        self.verbose = verbose

        start_time = time.time()
        self._print("Fitting model... ")

        # Step 1: Normalize the URM if required
        if normalize_matrix:
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)

        # Step 2: Compute similarity matrix using the user-item matrix
        similarity = Compute_Similarity(self.URM_train, shrink=0, topK=topK, normalize=False, similarity="cosine")
        mat = similarity.compute_similarity()

        # Step 3: Apply Top-K filtering to the similarity matrix (keep only top-K neighbors)
        mat = similarityMatrixTopK(mat, k=topK, verbose=False)
        self._print(f"Matrice sparsa (top-K): {mat.shape}")

        # Step 4: Operazioni sparse - Utilizziamo PyTorch per il calcolo delle matrici sparse
        mat_torch = torch.sparse.FloatTensor(torch.LongTensor(mat.nonzero()), torch.FloatTensor(mat.data), mat.shape)

        # Step 5: Graham matrix
        grahm_matrix = mat_torch.T @ mat_torch  # Transpose and multiply to get the Graham matrix
        item_popularity = np.ediff1d(self.URM_train.tocsc().indptr)

        # Step 6: Aggiungiamo la popolarità alla diagonale della matrice Graham
        grahm_matrix = grahm_matrix.to_dense()  # Convert sparse matrix to dense for easier manipulation
        diag_indices = torch.arange(grahm_matrix.shape[0])
        grahm_matrix[diag_indices, diag_indices] += torch.tensor(item_popularity)

        # Step 7: Calcoliamo l'inversa della matrice Graham
        P = torch.inverse(grahm_matrix)  # Matrice inversa

        # Step 8: Calcoliamo la matrice B
        B = P / (-torch.diag(P).reshape(-1, 1))  # Divide each row by its diagonal
        B[diag_indices, diag_indices] = 0  # Impostiamo la diagonale a zero per evitare effetti di auto-interazione

        self.B = B

        # Step 9: Calcoliamo la matrice delle previsioni
        self.pred = mat_torch @ B  # Moltiplichiamo la matrice sparsa per la matrice B (ancora sparsa)

        # Print fitting time
        new_time_value, new_time_unit = self.seconds_to_biggest_unit(time.time() - start_time)
        self._print("Fitting model... done in {:.2f} {}".format(new_time_value, new_time_unit))

    def _compute_score_W_dense(self, user_id_array, items_to_compute=None):
        """
        Calcola i punteggi utilizzando la matrice B in formato denso
        """
        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)

        return item_scores

    def _is_content_sparse_check(self, matrix):
        """
        Verifica se la matrice è sparsa.
        """
        if self.sparse_threshold_quota is None:
            return False

        if sps.issparse(matrix):
            nonzero = matrix.nnz
        else:
            nonzero = np.count_nonzero(matrix)

        return nonzero / (matrix.shape[0]**2) <= self.sparse_threshold_quota

    def load_model(self, folder_path, file_name=None):
        super(EASE_R_PyTorch, self).load_model(folder_path, file_name=file_name)
        if not torch.sparse.is_sparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense

    def seconds_to_biggest_unit(self, seconds):
        """
        Convert seconds into the largest time unit possible (e.g., hours, minutes).
        """
        if seconds < 60:
            return seconds, "seconds"
        elif seconds < 3600:
            return seconds // 60, "minutes"
        elif seconds < 86400:
            return seconds // 3600, "hours"
        else:
            return seconds // 86400, "days"
