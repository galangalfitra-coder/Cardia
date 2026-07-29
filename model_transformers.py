"""Custom transformer classes, dipisah dari main.py supaya pickle bisa
di-load dari app.py tanpa perlu import seluruh main.py (yang menjalankan
training saat diimpor). WAJIB ada file ini di folder yang sama dengan
app.py dan best_catboost_model.pkl, termasuk saat di-hosting."""

from sklearn.base import BaseEstimator, TransformerMixin


class CholesterolZeroImputer(BaseEstimator, TransformerMixin):
    """Mengganti nilai cholesterol = 0 (missing value tersembunyi) dengan
    median cholesterol yang valid (> 0). Median dihitung HANYA dari data
    training saat fit() dipanggil, untuk mencegah data leakage."""

    def __init__(self, column="cholesterol"):
        self.column = column
        self.median_ = None

    def fit(self, X, y=None):
        X = X.copy()
        valid = X.loc[X[self.column] > 0, self.column].astype(float)
        self.median_ = float(valid.median())
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].astype(float)
        X.loc[X[self.column] == 0, self.column] = self.median_
        return X
