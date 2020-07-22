import time

import numpy as np
import pandas as pd

from .fast_methods import _compute_val_metrics
from .fast_methods import _initialization
from .fast_methods import _run_epoch
from .fast_methods import _shuffle
from .utils import timer


class SVD:
    """Implements Simon Funk SVD algorithm engineered during the Netflix Prize.

    Attributes:
        lr (float): learning rate.
        reg (float): regularization factor.
        n_epochs (int): number of SGD iterations.
        n_factors (int): number of latent factors.
        global_mean (float): ratings arithmetic mean.
        pu (numpy array): users latent factor matrix.
        qi (numpy array): items latent factor matrix.
        bu (numpy array): users biases vector.
        bi (numpy array): items biases vector.
        early_stopping (boolean): whether or not to stop training based on a
            validation monitoring.
        shuffle (boolean): whether or not to shuffle data before each epoch.

    """

    def __init__(self, learning_rate=.005, regularization=0.02, n_epochs=20,
                 n_factors=100, min_rating=1, max_rating=5):

        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.early_stopping = False
        self.shuffle = False
        self.global_mean = np.nan
        self.metrics_ = None
        self.min_delta_ = 0.001

    def _preprocess_data(self, X, train=True):
        """Maps users and items ids to indexes and returns a numpy array.

        Args:
            X (pandas DataFrame): dataset.
            train (boolean): whether or not X is the training set or the
                validation set.

        Returns:
            X (numpy array): mapped dataset.
        """
        X = X.copy()

        if train:
            u_ids = X['u_id'].unique().tolist()
            i_ids = X['i_id'].unique().tolist()

            self.user_dict = dict(zip(u_ids, list(range(len(u_ids)))))
            self.item_dict = dict(zip(i_ids, list(range(len(i_ids)))))

        X['u_id'] = X['u_id'].map(self.user_dict)
        X['i_id'] = X['i_id'].map(self.item_dict)

        # Tag unknown users/items with -1 (when val)
        X.fillna(-1, inplace=True)

        X['u_id'] = X['u_id'].astype(np.int32)
        X['i_id'] = X['i_id'].astype(np.int32)

        return X[['u_id', 'i_id', 'rating']].values

    def _sgd(self, X, X_val):
        """Performs SGD algorithm, learns model weights.

        Args:
            X (numpy array): training set, first column must contains users
                indexes, second one items indexes, and third one ratings.
            X_val (numpy array or `None`): validation set with same structure
                as X.
        """
        n_user = len(np.unique(X[:, 0]))
        n_item = len(np.unique(X[:, 1]))

        pu, qi, bu, bi = _initialization(n_user, n_item, self.n_factors)

        # Run SGD
        for epoch_ix in range(self.n_epochs):
            start = self._on_epoch_begin(epoch_ix)

            if self.shuffle:
                X = _shuffle(X)

            pu, qi, bu, bi = _run_epoch(X, pu, qi, bu, bi, self.global_mean,
                                        self.n_factors, self.lr, self.reg)

            if X_val is not None:
                self.metrics_[epoch_ix, :] = _compute_val_metrics(X_val, pu, qi, bu, bi,
                                                                  self.global_mean,
                                                                  self.n_factors)
                self._on_epoch_end(start,
                                   self.metrics_[epoch_ix, 0],
                                   self.metrics_[epoch_ix, 1],
                                   self.metrics_[epoch_ix, 2])

                if self.early_stopping:
                    if self._early_stopping(self.metrics_[:, 1], epoch_ix, self.min_delta_):
                        break

            else:
                self._on_epoch_end(start)

        self.pu = pu
        self.qi = qi
        self.bu = bu
        self.bi = bi

    @timer(text='\nTraining took ')
    def fit(self, X, X_val=None, early_stopping=False, shuffle=False, min_delta=0.001):
        """Learns model weights.

        Args:
            X (pandas DataFrame): training set, must have `u_id` for user id,
                `i_id` for item id and `rating` columns.
            X_val (pandas DataFrame, defaults to `None`): validation set with
                same structure as X.
            early_stopping (boolean): whether or not to stop training based on
                a validation monitoring.
            shuffle (boolean): whether or not to shuffle the training set
                before each epoch.
            min_delta (float, defaults to .001): minimun delta to arg for an
                improvement.

        Returns:
            self (SVD object): the current fitted object.
        """
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.min_delta_ = min_delta
        print('Preprocessing data...\n')
        X = self._preprocess_data(X)
        if X_val is not None:
            self.metrics_ = np.zeros((self.n_epochs, 3), dtype=np.float)
            X_val = self._preprocess_data(X_val, train=False)

        self.global_mean = np.mean(X[:, 2])
        self._sgd(X, X_val)

        return self

    def predict_pair(self, u_id, i_id, clip=True):
        """Returns the model rating prediction for a given user/item pair.

        Args:
            u_id (int): a user id.
            i_id (int): an item id.
            clip (boolean, default is `True`): whether to clip the prediction
                or not.

        Returns:
            pred (float): the estimated rating for the given user/item pair.
        """
        user_known, item_known = False, False
        pred = self.global_mean

        if u_id in self.user_dict:
            user_known = True
            u_ix = self.user_dict[u_id]
            pred += self.bu[u_ix]

        if i_id in self.item_dict:
            item_known = True
            i_ix = self.item_dict[i_id]
            pred += self.bi[i_ix]

        if user_known and item_known:
            pred += np.dot(self.pu[u_ix], self.qi[i_ix])

        if clip:
            pred = self.max_rating if pred > self.max_rating else pred
            pred = self.min_rating if pred < self.min_rating else pred

        return pred

    def predict(self, X):
        """Returns estimated ratings of several given user/item pairs.

        Args:
            X (pandas DataFrame): storing all user/item pairs we want to
                predict the ratings. Must contains columns labeled `u_id` and
                `i_id`.

        Returns:
            predictions: list, storing all predictions of the given user/item
                pairs.
        """
        predictions = []

        for u_id, i_id in zip(X['u_id'], X['i_id']):
            predictions.append(self.predict_pair(u_id, i_id))

        return predictions

    def _early_stopping(self, list_val_rmse, epoch_idx, min_delta):
        """Returns True if validation rmse is not improving.

        Last rmse (plus `min_delta`) is compared with the second to last.

        Agrs:
            list_val_rmse (list): ordered validation RMSEs.
            min_delta (float): minimun delta to arg for an improvement.

        Returns:
            (boolean): whether or not to stop training.
        """
        if epoch_idx > 0:
            if list_val_rmse[epoch_idx] + min_delta > list_val_rmse[epoch_idx-1]:
                self.metrics_ = self.metrics_[:(epoch_idx+1), :]
                return True
        return False

    def _on_epoch_begin(self, epoch_ix):
        """Displays epoch starting log and returns its starting time.

        Args:
            epoch_ix: integer, epoch index.

        Returns:
            start (float): starting time of the current epoch.
        """
        start = time.time()
        end = '  | ' if epoch_ix < 9 else ' | '
        print('Epoch {}/{}'.format(epoch_ix + 1, self.n_epochs), end=end)

        return start

    def _on_epoch_end(self, start, val_loss=None, val_rmse=None, val_mae=None):
        """
        Displays epoch ending log. If self.verbose compute and display
        validation metrics (loss/rmse/mae).

        # Arguments
            start (float): starting time of the current epoch.
            val_loss: float, validation loss
            val_rmse: float, validation rmse
            val_mae: float, validation mae
        """
        end = time.time()

        if val_loss is not None:
            print('val_loss: {:.2f}'.format(val_loss), end=' - ')
            print('val_rmse: {:.2f}'.format(val_rmse), end=' - ')
            print('val_mae: {:.2f}'.format(val_mae), end=' - ')

        print('took {:.1f} sec'.format(end - start))

    def get_val_metrics(self):
        """Get validation metrics

        Returns:
            a Pandas DataFrame
        """
        if isinstance(self.metrics_, np.ndarray) and (self.metrics_.shape[1] == 3):
            return pd.DataFrame(self.metrics_, columns=["Loss", "RMSE", "MAE"])
