import numpy as np

from numba import njit


@njit
def _shuffle(X):
    np.random.shuffle(X)
    return X


@njit
def _initialization(n_user, n_item, n_factors):
    """Initializes biases and latent factor matrixes.

    Args:
        n_user (int): number of different users.
        n_item (int): number of different items.
        n_factors (int): number of factors.

    Returns:
        pu (numpy array): users latent factor matrix.
        qi (numpy array): items latent factor matrix.
        bu (numpy array): users biases vector.
        bi (numpy array): items biases vector.
    """
    pu = np.random.normal(0, .1, (n_user, n_factors))
    qi = np.random.normal(0, .1, (n_item, n_factors))

    bu = np.zeros(n_user)
    bi = np.zeros(n_item)

    return pu, qi, bu, bi


@njit
def _run_epoch(X, pu, qi, bu, bi, global_mean, n_factors, lr, reg):
    """Runs an epoch, updating model weights (pu, qi, bu, bi).

    Args:
        X (numpy array): training set.
        pu (numpy array): users latent factor matrix.
        qi (numpy array): items latent factor matrix.
        bu (numpy array): users biases vector.
        bi (numpy array): items biases vector.
        global_mean (float): ratings arithmetic mean.
        n_factors (int): number of latent factors.
        lr (float): learning rate.
        reg (float): regularization factor.

    Returns:
        pu (numpy array): users latent factor matrix updated.
        qi (numpy array): items latent factor matrix updated.
        bu (numpy array): users biases vector updated.
        bi (numpy array): items biases vector updated.
    """
    for i in range(X.shape[0]):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

        # Predict current rating
        pred = global_mean + bu[user] + bi[item]

        for factor in range(n_factors):
            pred += pu[user, factor] * qi[item, factor]

        err = rating - pred

        # Update biases
        bu[user] += lr * (err - reg * bu[user])
        bi[item] += lr * (err - reg * bi[item])

        # Update latent factors
        for factor in range(n_factors):
            puf = pu[user, factor]
            qif = qi[item, factor]

            pu[user, factor] += lr * (err * qif - reg * puf)
            qi[item, factor] += lr * (err * puf - reg * qif)

    return pu, qi, bu, bi


@njit
def _compute_val_metrics(X_val, pu, qi, bu, bi, global_mean, n_factors):
    """Computes validation metrics (loss, rmse, and mae).

    Args:
        X_val (numpy array): validation set.
        pu (numpy array): users latent factor matrix.
        qi (numpy array): items latent factor matrix.
        bu (numpy array): users biases vector.
        bi (numpy array): items biases vector.
        global_mean (float): ratings arithmetic mean.
        n_factors (int): number of latent factors.

    Returns:
        (tuple of floats): validation loss, rmse and mae.
    """
    residuals = []

    for i in range(X_val.shape[0]):
        user, item, rating = int(X_val[i, 0]), int(X_val[i, 1]), X_val[i, 2]
        pred = global_mean

        if user > -1:
            pred += bu[user]

        if item > -1:
            pred += bi[item]

        if (user > -1) and (item > -1):
            for factor in range(n_factors):
                pred += pu[user, factor] * qi[item, factor]

        residuals.append(rating - pred)

    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()

    return loss, rmse, mae
