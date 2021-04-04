import numpy as np
import pandas as pd
import time

from .fast_methods import _compute_val_metrics
from .fast_methods import _initialization
from .fast_methods import _run_epoch
from .fast_methods import _shuffle
from .utils import _timer


__all__ = ['SVD']


class SVD:
    """Implements Simon Funk famous SVD algorithm.

    Parameters
    ----------
    lr : float, default=.005
        Learning rate.
    reg : float, default=.02
        L2 regularization factor.
    n_epochs : int, default=20
        Number of SGD iterations.
    n_factors : int, default=100
        Number of latent factors.
    early_stopping : bool, default=False
        Whether or not to stop training based on a validation monitoring.
    shuffle : bool, default=False
        Whether or not to shuffle the training set before each epoch.
    min_delta : float, default=.001
        Minimun delta to argue for an improvement.
    min_rating : int, default=1
        Minimum value a rating should be clipped to at inference time.
    max_rating : int, default=5
        Maximum value a rating should be clipped to at inference time.

    Attributes
    ----------
    user_mapping_ : dict
        Maps user ids to their indexes.
    item_mapping_ : dict
        Maps item ids to their indexes.
    global_mean_ : float
        Ratings arithmetic mean.
    pu_ : numpy.array
        User latent factors matrix.
    qi_ : numpy.array
        Item latent factors matrix.
    bu_ : numpy.array
        User biases vector.
    bi_ : numpy.array
        Item biases vector.
    metrics_ : pandas.DataFrame
        Validation metrics at each epoch. Column names are 'Loss', 'RMSE', and
        'MAE'.

    """

    def __init__(self, lr=.005, reg=.02, n_epochs=20, n_factors=100,
                 early_stopping=False, shuffle=False, min_delta=.001,
                 min_rating=1, max_rating=5):

        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.min_delta = min_delta
        self.min_rating = min_rating
        self.max_rating = max_rating

    @_timer(text='\nTraining took ')
    def fit(self, X, X_val=None):
        """Learns model weights from input data.

        Parameters
        ----------
        X : pandas.DataFrame
            Training set, must have 'u_id' for user ids, 'i_id' for item ids,
            and 'rating' column names.
        X_val : pandas.DataFrame, default=None
            Validation set with the same column structure as X.

        Returns
        -------
        self : SVD object
            The current fitted object.
        """
        X = self._preprocess_data(X)

        if X_val is not None:
            X_val = self._preprocess_data(X_val, train=False, verbose=False)
            self._init_metrics()

        self.global_mean_ = np.mean(X[:, 2])
        self._run_sgd(X, X_val)

        return self

    def _preprocess_data(self, X, train=True, verbose=True):
        """Maps user and item ids to their indexes.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset, must have 'u_id' for user ids, 'i_id' for item ids, and
            'rating' column names.
        train : boolean
            Whether or not X is the training set or the validation set.

        Returns
        -------
        X : numpy.array
            Mapped dataset.
        """
        print('Preprocessing data...\n')
        X = X.copy()

        if train:  # Mappings have to be created
            user_ids = X['u_id'].unique().tolist()
            item_ids = X['i_id'].unique().tolist()

            n_users = len(user_ids)
            n_items = len(item_ids)

            user_idx = range(n_users)
            item_idx = range(n_items)

            self.user_mapping_ = dict(zip(user_ids, user_idx))
            self.item_mapping_ = dict(zip(item_ids, item_idx))

        X['u_id'] = X['u_id'].map(self.user_mapping_)
        X['i_id'] = X['i_id'].map(self.item_mapping_)

        # Tag validation set unknown users/items with -1 (enables
        # `fast_methods._compute_val_metrics` detecting them)
        X.fillna(-1, inplace=True)

        X['u_id'] = X['u_id'].astype(np.int32)
        X['i_id'] = X['i_id'].astype(np.int32)

        return X[['u_id', 'i_id', 'rating']].values

    def _init_metrics(self):
        metrics = np.zeros((self.n_epochs, 3), dtype=float)
        self.metrics_ = pd.DataFrame(metrics, columns=['Loss', 'RMSE', 'MAE'])

    def _run_sgd(self, X, X_val):
        """Runs SGD algorithm, learning model weights.

        Parameters
        ----------
        X : numpy.array
            Training set, first column must be user indexes, second one item
            indexes, and third one ratings.
        X_val : numpy.array or None
            Validation set with the same structure as X.
        """
        n_users = len(np.unique(X[:, 0]))
        n_items = len(np.unique(X[:, 1]))

        bu, bi, pu, qi = _initialization(n_users, n_items, self.n_factors)

        # Run SGD
        for epoch_ix in range(self.n_epochs):
            start = self._on_epoch_begin(epoch_ix)

            if self.shuffle:
                X = _shuffle(X)

            bu, bi, pu, qi = _run_epoch(X, bu, bi, pu, qi, self.global_mean_,
                                        self.n_factors, self.lr, self.reg)

            if X_val is not None:
                self.metrics_.loc[epoch_ix, :] = _compute_val_metrics(
                                                     X_val, bu, bi, pu, qi,
                                                     self.global_mean_,
                                                     self.n_factors
                                                 )
                self._on_epoch_end(start,
                                   self.metrics_.loc[epoch_ix, 'Loss'],
                                   self.metrics_.loc[epoch_ix, 'RMSE'],
                                   self.metrics_.loc[epoch_ix, 'MAE'])

                if self.early_stopping:
                    val_rmse = self.metrics_['RMSE'].tolist()
                    if self._early_stopping(val_rmse, epoch_ix,
                                            self.min_delta):
                        break

            else:
                self._on_epoch_end(start)

        self.bu_ = bu
        self.bi_ = bi
        self.pu_ = pu
        self.qi_ = qi

    def predict(self, X, clip=True):
        """Returns estimated ratings of several given user/item pairs.

        Parameters
        ----------
        X : pandas.DataFrame
            Storing all user/item pairs we want to predict the ratings. Must
            contains columns labeled 'u_id' and 'i_id'.
        clip : bool, default=True
            Whether to clip the predictions or not.

        Returns
        -------
        predictions : list
            Predictions belonging to the input user/item pairs.
        """
        return [
            self.predict_pair(u_id, i_id, clip)
            for u_id, i_id in zip(X['u_id'], X['i_id'])
        ]

    def predict_pair(self, u_id, i_id, clip=True):
        """Returns the model rating prediction for a given user/item pair.

        Parameters
        ----------
        u_id : int
            A user id.
        i_id : int
            An item id.
        clip : bool, default=True
            Whether to clip the prediction or not.

        Returns
        -------
        pred : float
            The estimated rating for the given user/item pair.
        """
        user_known, item_known = False, False
        pred = self.global_mean_

        if u_id in self.user_mapping_:
            user_known = True
            u_ix = self.user_mapping_[u_id]
            pred += self.bu_[u_ix]

        if i_id in self.item_mapping_:
            item_known = True
            i_ix = self.item_mapping_[i_id]
            pred += self.bi_[i_ix]

        if user_known and item_known:
            pred += np.dot(self.pu_[u_ix], self.qi_[i_ix])

        if clip:
            pred = self.max_rating if pred > self.max_rating else pred
            pred = self.min_rating if pred < self.min_rating else pred

        return pred

    def _early_stopping(self, val_rmse, epoch_idx, min_delta):
        """Returns True if validation rmse is not improving.

        Last rmse (plus `min_delta`) is compared with the second to last.

        Parameters
        ----------
        val_rmse : list
            Validation RMSEs.
        min_delta : float
            Minimun delta to argue for an improvement.

        Returns
        -------
        early_stopping : bool
            Whether to stop training or not.
        """
        if epoch_idx > 0:
            if val_rmse[epoch_idx] + min_delta > val_rmse[epoch_idx-1]:
                self.metrics_ = self.metrics_.loc[:(epoch_idx+1), :]
                return True
        return False

    def _on_epoch_begin(self, epoch_ix):
        """Displays epoch starting log and returns its starting time.

        Parameters
        ----------
        epoch_ix : int
            Epoch index.

        Returns
        -------
        start : float
            Starting time of the current epoch.
        """
        start = time.time()
        end = '  | ' if epoch_ix < 9 else ' | '
        print('Epoch {}/{}'.format(epoch_ix + 1, self.n_epochs), end=end)

        return start

    def _on_epoch_end(self, start, val_loss=None, val_rmse=None, val_mae=None):
        """Displays epoch ending log.

        If self.verbose, computes and displays validation metrics (loss, rmse,
        and mae).

        Parameters
        ----------
        start : float
            Starting time of the current epoch.
        val_loss : float, default=None
            Validation loss.
        val_rmse : float, default=None
            Validation rmse.
        val_mae : float, default=None
            Validation mae.
        """
        end = time.time()

        if val_loss is not None:
            print(f'val_loss: {val_loss:.2f}', end=' - ')
            print(f'val_rmse: {val_rmse:.2f}', end=' - ')
            print(f'val_mae: {val_mae:.2f}', end=' - ')

        print(f'took {end - start:.1f} sec')
