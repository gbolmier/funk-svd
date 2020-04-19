import pandas as pd
import numpy as np

from funk_svd.dataset import fetch_ml_ratings
from funk_svd import SVD

from sklearn.metrics import mean_absolute_error

df = fetch_ml_ratings(variant='100k')

train = df.sample(frac=0.8, random_state=7)
val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = df.drop(train.index.tolist()).drop(val.index.tolist())

svd = SVD(learning_rate=0.001, regularization=0.005, n_epochs=100,
          n_factors=15, min_rating=1, max_rating=5)

df_matrix_original = svd.get_utility_matrix(df)
print ("Original Utility Matrix: \n", df_matrix_original.values)

# Getting all u_id and i_id combinations
df_user_item = pd.melt(df_matrix_original.reset_index(drop=False), id_vars='u_id')

svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

pred_test = svd.predict(test)
df_user_item["rating"] = svd.predict(df_user_item)

print ("Predicted Utility Matrix: \n", svd.get_utility_matrix(df_user_item).values)

mae = mean_absolute_error(test["rating"], pred_test)

print(f'Test MAE: {mae:.2f}')
