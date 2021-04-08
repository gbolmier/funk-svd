# :zap: funk-svd [![Build Status](https://img.shields.io/travis/gbolmier/funk-svd/master.svg?style=flat)](https://travis-ci.com/gbolmier/funk-svd) [![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](https://opensource.org/licenses/MIT)

`funk-svd` is a Python 3 library implementing a fast version of the famous SVD algorithm [popularized](http://sifter.org/simon/journal/20061211.html) by Simon Funk during the [Neflix Prize](http://en.wikipedia.org/wiki/Netflix_Prize) contest.

[`Numba`](http://numba.pydata.org/) is used to speed up our algorithm, enabling us to run over 10 times faster than [`Surprise`](http://surpriselib.com)'s Cython implementation (cf. [benchmark notebook](http://nbviewer.jupyter.org/github/gbolmier/funk-svd/blob/master/benchmark.ipynb)).

| Movielens 20M | RMSE   | MAE    | Time          |
|:--------------|:------:|:------:|--------------:|
| Surprise      |  0.88  |  0.68  | 10 min 40 sec |
| Funk-svd      |  0.88  |  0.68  |        42 sec |

## Installation

Run `pip install git+https://github.com/gbolmier/funk-svd` in your terminal.

## Contributing

All contributions, bug reports, bug fixes, enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the [contributor guide](CONTRIBUTING.md).

## Quick example

[run_experiment.py](run_experiment.py):

```python
>>> from funk_svd.dataset import fetch_ml_ratings
>>> from funk_svd import SVD

>>> from sklearn.metrics import mean_absolute_error


>>> df = fetch_ml_ratings(variant='100k')

>>> train = df.sample(frac=0.8, random_state=7)
>>> val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
>>> test = df.drop(train.index.tolist()).drop(val.index.tolist())

>>> svd = SVD(lr=0.001, reg=0.005, n_epochs=100, n_factors=15,
...           early_stopping=True, shuffle=False, min_rating=1, max_rating=5)

>>> svd.fit(X=train, X_val=val)
Preprocessing data...

Epoch 1/...

>>> pred = svd.predict(test)
>>> mae = mean_absolute_error(test['rating'], pred)

>>> print(f'Test MAE: {mae:.2f}')
Test MAE: 0.75

```

## Funk SVD for recommendation in a nutshell

We have a huge sparse matrix:

<a href="https://www.codecogs.com/eqnedit.php?latex=R&space;=&space;\begin{pmatrix}&space;{\color{Red}&space;?}&space;&&space;2&space;&&space;\cdots&space;&&space;{\color{Red}&space;?}&space;&&space;{\color{Red}&space;?}&space;\\&space;{\color{Red}&space;?}&space;&&space;{\color{Red}&space;?}&space;&&space;\cdots&space;&&space;{\color{Red}&space;?}&space;&&space;4.5&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\ddots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;3&space;&&space;{\color{Red}&space;?}&space;&&space;\cdots&space;&&space;{\color{Red}&space;?}&space;&&space;{\color{Red}&space;?}&space;\\&space;{\color{Red}&space;?}&space;&&space;{\color{Red}&space;?}&space;&&space;\cdots&space;&&space;5&space;&&space;{\color{Red}&space;?}&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R&space;=&space;\begin{pmatrix}&space;{\color{Red}&space;?}&space;&&space;2&space;&&space;\cdots&space;&&space;{\color{Red}&space;?}&space;&&space;{\color{Red}&space;?}&space;\\&space;{\color{Red}&space;?}&space;&&space;{\color{Red}&space;?}&space;&&space;\cdots&space;&&space;{\color{Red}&space;?}&space;&&space;4.5&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\ddots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;3&space;&&space;{\color{Red}&space;?}&space;&&space;\cdots&space;&&space;{\color{Red}&space;?}&space;&&space;{\color{Red}&space;?}&space;\\&space;{\color{Red}&space;?}&space;&&space;{\color{Red}&space;?}&space;&&space;\cdots&space;&&space;5&space;&&space;{\color{Red}&space;?}&space;\end{pmatrix}" title="R = \begin{pmatrix} {\color{Red} ?} & 2 & \cdots & {\color{Red} ?} & {\color{Red} ?} \\ {\color{Red} ?} & {\color{Red} ?} & \cdots & {\color{Red} ?} & 4.5 \\ \vdots & \ddots & \ddots & \ddots & \vdots \\ 3 & {\color{Red} ?} & \cdots & {\color{Red} ?} & {\color{Red} ?} \\ {\color{Red} ?} & {\color{Red} ?} & \cdots & 5 & {\color{Red} ?} \end{pmatrix}" /></a>

storing known ratings for a set of users and items:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;u&space;=&space;1,&space;...,&space;U" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;u&space;=&space;1,&space;...,&space;U" title="u = 1, ..., U" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i&space;=&space;1,&space;...,&space;I" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;i&space;=&space;1,&space;...,&space;I" title="i = 1, ..., I" /></a>

The idea is to estimate unknown ratings by factorizing the rating matrix into two smaller matrices representing user and item characteristics:

<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;=&space;\begin{pmatrix}&space;0.37&space;&&space;\cdots&space;&&space;0.69&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;1.08&space;&&space;\cdots&space;&&space;0.24&space;\end{pmatrix}&space;,&space;Q&space;=&space;\begin{pmatrix}&space;0.09&space;&&space;\cdots&space;&&space;\cdots&space;&&space;\cdots&space;&&space;0.46&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\ddots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;0.51&space;&&space;\cdots&space;&&space;\cdots&space;&&space;\cdots&space;&&space;0.72&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P&space;=&space;\begin{pmatrix}&space;0.37&space;&&space;\cdots&space;&&space;0.69&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;1.08&space;&&space;\cdots&space;&&space;0.24&space;\end{pmatrix}&space;,&space;Q&space;=&space;\begin{pmatrix}&space;0.09&space;&&space;\cdots&space;&&space;\cdots&space;&&space;\cdots&space;&&space;0.46&space;\\&space;\vdots&space;&&space;\ddots&space;&&space;\ddots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;0.51&space;&&space;\cdots&space;&&space;\cdots&space;&&space;\cdots&space;&&space;0.72&space;\end{pmatrix}" title="P = \begin{pmatrix} 0.37 & \cdots & 0.69 \\ \vdots & \ddots & \vdots \\ \vdots & \ddots & \vdots \\ \vdots & \ddots & \vdots \\ 1.08 & \cdots & 0.24 \end{pmatrix} , Q = \begin{pmatrix} 0.09 & \cdots & \cdots & \cdots & 0.46 \\ \vdots & \ddots & \ddots & \ddots & \vdots \\ 0.51 & \cdots & \cdots & \cdots & 0.72 \end{pmatrix}" /></a>

We call these two matrices users and items latent factors. Then, by applying the dot product between both matrices we can reconstruct our rating matrix. The trick is that the empty values will now contain estimated ratings.

In order to get more accurate results, the global average rating as well as the user and item biases are used in addition:

<a href="https://www.codecogs.com/eqnedit.php?latex=\bar{r}&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{N}&space;K_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bar{r}&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{N}&space;K_{i}" title="\bar{r} = \frac{1}{N} \sum_{i=1}^{N} K_{i}" /></a>

where K stands for known ratings.

<a href="https://www.codecogs.com/eqnedit.php?latex=bu&space;=&space;\begin{pmatrix}&space;0.35&space;&&space;\cdots&space;&&space;0.07&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?bu&space;=&space;\begin{pmatrix}&space;0.35&space;&&space;\cdots&space;&&space;0.07&space;\end{pmatrix}" title="bu = \begin{pmatrix} 0.35 & \cdots & 0.07 \end{pmatrix}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=bi&space;=&space;\begin{pmatrix}&space;0.16&space;&&space;\cdots&space;&&space;0.40&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?bi&space;=&space;\begin{pmatrix}&space;0.16&space;&&space;\cdots&space;&&space;0.40&space;\end{pmatrix}" title="bi = \begin{pmatrix} 0.16 & \cdots & 0.40 \end{pmatrix}" /></a>

Then, we can estimate any rating by applying:

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{r}_{u,&space;i}&space;=&space;\bar{r}&space;&plus;&space;bu_{u}&space;&plus;&space;bi_{i}&space;&plus;&space;\sum_{f=1}^{F}&space;P_{u,&space;f}&space;*&space;Q_{i,&space;f}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{r}_{u,&space;i}&space;=&space;\bar{r}&space;&plus;&space;bu_{u}&space;&plus;&space;bi_{i}&space;&plus;&space;\sum_{f=1}^{F}&space;P_{u,&space;f}&space;*&space;Q_{i,&space;f}" title="\hat{r}_{u, i} = \bar{r} + bu_{u} + bi_{i} + \sum_{f=1}^{F} P_{u, f} * Q_{i, f}" /></a>

The learning step consists in performing the SGD algorithm where for each known rating the biases and latent factors are updated as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=err&space;=&space;r&space;-&space;\hat{r}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?err&space;=&space;r&space;-&space;\hat{r}" title="err = r - \hat{r}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=bu_{u}&space;=&space;bu_{u}&space;&plus;&space;\alpha&space;*&space;(err&space;-&space;\lambda&space;*&space;bu_{u})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?bu_{u}&space;=&space;bu_{u}&space;&plus;&space;\alpha&space;*&space;(err&space;-&space;\lambda&space;*&space;bu_{u})" title="bu_{u} = bu_{u} + \alpha * (err - \lambda * bu_{u})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=bi_{i}&space;=&space;bi_{i}&space;&plus;&space;\alpha&space;*&space;(err&space;-&space;\lambda&space;*&space;bi_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?bi_{i}&space;=&space;bi_{i}&space;&plus;&space;\alpha&space;*&space;(err&space;-&space;\lambda&space;*&space;bi_{i})" title="bi_{i} = bi_{i} + \alpha * (err - \lambda * bi_{i})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=P_{u,&space;f}&space;=&space;P_{u,&space;f}&space;&plus;&space;\alpha&space;*&space;(err&space;*&space;Q_{i,&space;f}&space;-&space;\lambda&space;*&space;P_{u,&space;f})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P_{u,&space;f}&space;=&space;P_{u,&space;f}&space;&plus;&space;\alpha&space;*&space;(err&space;*&space;Q_{i,&space;f}&space;-&space;\lambda&space;*&space;P_{u,&space;f})" title="P_{u, f} = P_{u, f} + \alpha * (err * Q_{i, f} - \lambda * P_{u, f})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=Q_{i,&space;f}&space;=&space;Q_{i,&space;f}&space;&plus;&space;\alpha&space;*&space;(err&space;*&space;P_{u,&space;f}&space;-&space;\lambda&space;*&space;Q_{i,&space;f})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_{i,&space;f}&space;=&space;Q_{i,&space;f}&space;&plus;&space;\alpha&space;*&space;(err&space;*&space;P_{u,&space;f}&space;-&space;\lambda&space;*&space;Q_{i,&space;f})" title="Q_{i, f} = Q_{i, f} + \alpha * (err * P_{u, f} - \lambda * Q_{i, f})" /></a>

where alpha is the learning rate and lambda is the regularization term.

## References

- [Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Matrix completion](https://en.wikipedia.org/wiki/Matrix_completion)
- [Matrix factorization (recommender systems)](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
- [Recommender Systems Handbook](https://www.cse.iitk.ac.in/users/nsrivast/HCC/Recommender_systems_handbook.pdf)
- [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- [Surprise library for recommender systems](http://surpriselib.com/)

## License

MIT license, [see here](LICENSE).
