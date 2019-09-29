import pandas as pd
import numpy as np

from funk_svd.dataset import fetch_ml_ratings
from funk_svd import SVD

from sklearn.metrics import mean_absolute_error


df = fetch_ml_ratings(variant='20m')

train = df.sample(frac=0.8, random_state=7)
val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = df.drop(train.index.tolist()).drop(val.index.tolist())

svd = SVD(learning_rate=0.001, regularization=0.005, n_epochs=100,
          n_factors=15, min_rating=1, max_rating=5)

svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

pred = svd.predict(test)
mae = mean_absolute_error(test["rating"], pred)

print("Test MAE: {:.2f}".format(mae))
