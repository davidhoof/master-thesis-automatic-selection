import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class RandomForestPipeline:

    @staticmethod
    def __create_random_grid():
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
        # Create the random grid
        return {'rf__n_estimators': n_estimators,
                'rf__max_depth': max_depth,
                'rf__min_samples_split': min_samples_split,
                'rf__min_samples_leaf': min_samples_leaf,
                'rf__bootstrap': bootstrap}

    def __init__(self, features, random_grid=None, n_iter=100, cv=3,
                 verbose=2, random_state=42, n_jobs=-1):

        self.features = features
        self.random_grid = RandomForestPipeline.__create_random_grid()

        if random_grid is not None:
            self.random_grid = random_grid

        self.rf = Pipeline([
            ("sc", StandardScaler()),
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

    def plot_importances(self, X, y, figsize=(20, 7),c=None):
        if self.regression_model is None:
            raise ValueError("Model must be fitted before usage")
            
        # print("Hello")
        
        labels={
            'finetune_dataset': 'Finetuning-Datensatz',
             'pretrained_dataset': 'Pretraining-Datensatz',
             'model': 'Architektur',
             'checkpoint': 'Checkpoint',
             'sparsity': 'Sparsity',
             'variance_entropy': 'Entropie-Varianz',
             'variance_entropy_clean': 'Gereinigte Entropie-Varianz',
             'weighted_sparsity': 'Gewichtete Sparsity',
             'weighted_variance_entropy': 'Gewichtete Entropie-Varianz',
             'weighted_variance_entropy_clean': 'Gewichtete gereinigte Entropie-Varianz',
             'architecture_size_Conv2d': 'Architekturgröße (Conv2d)',
             'architecture_size_BatchNorm2d': 'Architekturgröße (BatchNorm2d)',
             'architecture_size_ReLU': 'Architekturgröße (ReLU)',
             'architecture_size_MaxPool2d': 'Architekturgröße (MaxPool)',
             'architecture_size_Flatten': 'Architekturgröße (Flatten)',
             'architecture_size_Linear': 'Architekturgröße (Linear)',
             'architecture_size': 'Architekturgröße (gesamt)',
             'kmeans_latent_space_silhouette_score': 'Latent-Space Silhouettenkoeffizienten',
             'dbscan_latent_space_silhouette_score': 'Latent-Space Silhouettenkoeffizienten',
             'difference': 'Transfer-Learning-Performanz',
             'architecture_size_AdaptiveAvgPool2d': 'Architekturgröße (ApdaptiveAvgPool2d)',
             'architecture_size_AvgPool2d': 'Architekturgröße (AvgPool2d)',
             'architecture_size_Dropout': 'Architekturgröße (Dropout)',
             'model_lowres_densenet121': 'Architektur (DenseNet-121)',
             'model_lowres_resnet50': 'Architektur (ResNet-50)',
             'model_lowres_resnet9': 'Architektur (ResNet-9)',
             'model_lowres_vgg16_bn': 'Architektur (VGG-16-bn)',
             'pretrained_dataset_cifar10(100)': 'Pretrainig-Datensatz (CIFAR-10)',
             'pretrained_dataset_cifar100(100)': 'Pretrainig-Datensatz (CIFAR-100)',
             'pretrained_dataset_grocerystore(100)': 'Pretrainig-Datensatz (GroceryStore)',
             'pretrained_dataset_svhn(100)': 'Pretrainig-Datensatz (SVHN)',
             'pretrained_dataset_tinyimagenet(100)': 'Pretrainig-Datensatz (TinyImageNet)'
        }

        importances = self.regression_model['rf'].feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.regression_model['rf'].estimators_], axis=0)
        
        

        forest_importances = pd.Series(importances, index=self.features).sort_values(ascending=False)    
        
        std_dict = dict(zip(self.features,std))
        std_sorted = np.array([std_dict[importance_] for importance_ in forest_importances.index])
        forest_importances=forest_importances.rename(labels) 
        
        # print(forest_importances)

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        
        if c!=None and c<len(forest_importances):
            forest_importances[:c].plot.bar(yerr=std_sorted[:c], ax=ax[0])
        else:        
            forest_importances.plot.bar(yerr=std_sorted, ax=ax[0])
        
        ax[0].set_title("Merkmalsausprägungen unter Verwendung von MDI")
        ax[0].set_ylabel("Mittlerer Verlust an Unreinheit")
        # print(ax[0].get_xticklabels())
        ax[0].set_xticks(ax[0].get_xticks(), ax[0].get_xticklabels(),rotation=45, ha="right")

        result = permutation_importance(
            self.regression_model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        forest_importances = pd.Series(result.importances_mean, index=self.features).sort_values(ascending=False)
        std_dict=dict(zip(self.features,result.importances_std))
        
        std_sorted= np.array([std_dict[importance_] for importance_ in forest_importances.index])
        
        forest_importances = forest_importances.rename(labels)
        
        if c!=None and c<len(forest_importances):            
            forest_importances[:c].plot.bar(yerr=std_sorted[:c], ax=ax[1])
        else:
            forest_importances.plot.bar(yerr=std_sorted, ax=ax[1])
            
        ax[1].set_title("Bedeutung von Merkmalen durch Permutation im vollständigen Modell")
        ax[1].set_ylabel("Mittlere Abnahme der Genauigkeit")
        ax[1].set_xticks(ax[1].get_xticks(), ax[1].get_xticklabels(),rotation=45, ha="right")
        # fig.tight_layout()
        # plt.show()
        return fig

    def score(self, X, y):
        return self.regression_model.score(X, y)
