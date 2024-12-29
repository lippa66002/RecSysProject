import torch
import torch.nn.functional as F
from torch import linalg as LA
from sklearn.preprocessing import normalize
import numpy as np
import time
import scipy.sparse as sps
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import similarityMatrixTopK, check_matrix
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from torch.sparse import mm
from torch.linalg import solve

class EASE_R_Filo(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "EASE_R_Filo"

    def __init__(self, URM_train, sparse_threshold_quota=None, verbose=True, device='cuda'):
        super(EASE_R_Filo, self).__init__(URM_train, verbose=verbose)
        self.sparse_threshold_quota = sparse_threshold_quota
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.URM_train = check_matrix(URM_train, format='csr')

    def fit(self, topK=None, l2_norm=1e3, normalize_matrix=False):

        torch.cuda.empty_cache()
        start_time = time.time()
        self._print("Fitting model... ")

        if normalize_matrix:
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)
            self.URM_train = sps.csr_matrix(self.URM_train)

        # Convert to sparse tensor
        urm_train_torch = self._convert_sparse_matrix_to_torch(self.URM_train)

        # Compute sparse Gram matrix
        gram_matrix = mm(urm_train_torch.T, urm_train_torch)

        # Add l2 regularization
        diag_indices = torch.arange(gram_matrix.size(0), device=self.device)
        gram_matrix = gram_matrix.to_dense()
        gram_matrix[diag_indices, diag_indices] += l2_norm

        # Solve instead of invert
        identity_matrix = torch.eye(gram_matrix.size(0), device=self.device, dtype=torch.float16)
        P = solve(gram_matrix, identity_matrix)

        # Compute B matrix
        B = P / (-torch.diag(P).view(-1, 1))
        B[diag_indices, diag_indices] = 0.0

        # Filter with topK
        B_np = B.cpu().numpy()
        if topK is not None:
            B_np = similarityMatrixTopK(B_np, k=topK, use_absolute_values=True, verbose=False)

        if self._is_content_sparse_check(B_np):
            self._print("Detected model matrix to be sparse, changing format.")
            self.W_sparse = check_matrix(B_np, format='csr', dtype=np.float16)
        else:
            self.W_sparse = check_matrix(B_np, format='npy', dtype=np.float16)
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)
        self._print("Fitting model... done in {:.2f} {}".format(new_time_value, new_time_unit))

    def _is_content_sparse_check(self, matrix):
        if self.sparse_threshold_quota is None:
            return False
        nonzero = matrix.nnz if sps.issparse(matrix) else np.count_nonzero(matrix)
        return nonzero / (matrix.shape[0]**2) <= self.sparse_threshold_quota

    def _convert_sparse_matrix_to_torch(self, sparse_matrix):
        coo = sparse_matrix.tocoo()
        indices = np.vstack((coo.row, coo.col))
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        values = torch.tensor(coo.data, dtype=torch.float16, device=self.device)
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def _compute_score_W_dense(self, user_id_array, items_to_compute=None):
        self._check_format()
        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = -np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float16) * np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)

        return item_scores

    def load_model(self, folder_path, file_name=None):
        super(EASE_R_Filo, self).load_model(folder_path, file_name=file_name)
        if not sps.issparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense
