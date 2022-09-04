import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class RandomForestPCAPipeline:

    def __create_random_grid(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=2, stop=200, num=30)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        
        #PCA
        n_components=range(2,len(self.features),2)
        # n_components=[1]
        
        # Create the random grid
        return {
                'pca__n_components': n_components,
                'rf__n_estimators': n_estimators,
                'rf__max_depth': max_depth,
                'rf__min_samples_split': min_samples_split,
                'rf__min_samples_leaf': min_samples_leaf,
                'rf__bootstrap': bootstrap}

    def __init__(self, features, random_grid=None, n_iter=100, cv=3,
                 verbose=2, random_state=42, n_jobs=-1):

        self.features = features
        self.random_grid = self.__create_random_grid()

        if random_grid is not None:
            self.random_grid = random_grid

        self.rf = Pipeline([
            ("sc", StandardScaler()),
            ('pca', PCA(n_components=8)),
            ("rf", RandomForestRegressor(max_depth=4, random_state=0))
        ])

        self.rf_random = RandomizedSearchCV(estimator=self.rf, param_distributions=self.random_grid, n_iter=n_iter, cv=cv,
                                            verbose=verbose, random_state=random_state, n_jobs=n_jobs)

        self.regression_model = None

    def fit(self, X, y):
        self.rf_random.fit(X, y)
        self.regression_model = self.rf_random.best_estimator_
        
    def predict(self, X):
        if self.regression_model is None:
            raise ValueError("Model must be fitted before usage")
        return self.regression_model.predict(X)        

    def plot_importances(self, X, y):
        if self.regression_model is None:
            raise ValueError("Model must be fitted before usage")

        importances = self.regression_model['rf'].feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.regression_model['rf'].estimators_], axis=0)

        forest_importances = pd.Series(importances, index=[f'PCA{c}' for c in range(self.rf_random.best_params_['pca__n_components'])]).sort_values(ascending=False)

        fig, ax = plt.subplots(1, 2, figsize=(20, 7))
        forest_importances.plot.bar(yerr=std, ax=ax[0])
        ax[0].set_title("Feature importances using MDI")
        ax[0].set_ylabel("Mean decrease in impurity")

        result = permutation_importance(
            self.regression_model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        forest_importances = pd.Series(result.importances_mean, index=self.features).sort_values(ascending=False)

        forest_importances.plot.bar(yerr=result.importances_std, ax=ax[1])
        ax[1].set_title("Feature importances using permutation on full model")
        ax[1].set_ylabel("Mean accuracy decrease")
        # fig.tight_layout()
        plt.show()

    def score(self, X, y):
        return self.regression_model.score(X, y)
