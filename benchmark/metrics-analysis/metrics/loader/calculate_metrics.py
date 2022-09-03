import os

from sklearn.cluster import KMeans, DBSCAN
from torchconvquality import measure_quality
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

import numpy as np
from metrics.utils import get_children

from metrics.utils import LayerHook
import data


class Metrics:
    """
    Superclass to implement different metrics for analysis of pretrained neural networks
    """

    def __init__(self):
        self.config = {}
        pass

    def calculate_metrics(self, model) -> dict:
        """
        Abstract method to define required parameters and return values
        :param model: model to calculate metrics from
        :return: dict of calculated metrics
        """
        return {}

    def add_parameter_to_config(self, key, value):
        self.config[key] = value

    def update_config(self, config):
        self.config.update(config)


class QualityMetrics(Metrics):
    """
    Quality metrics like sparsity and entropy variance. The class also calculates
    the weighted sparsity, entropy variance and the cleaned entropy variance
    """

    def __init__(self, sparsity_eps=0.1):
        super().__init__()
        self.sparsity_eps = sparsity_eps

    def calculate_metrics(self, model) -> dict:
        """
        Calculates quality metrics from given model
        :param model: model to calculate metrics from
        :return: dict of calculated quality metrics
        """
        info_dict = measure_quality(model, sparsity_eps=self.sparsity_eps)

        sparsity = []
        variance_entropy = []
        variance_entropy_clean = []
        weights = []

        for key_l, layer in info_dict.items():
            sparsity.append(layer['sparsity'])
            variance_entropy.append(layer['variance_entropy'])
            variance_entropy_clean.append(layer['variance_entropy_clean'])
            weights.append(layer['n'])

        return {
            "sparsity": np.average(sparsity),
            "variance_entropy": np.average(variance_entropy),
            "variance_entropy_clean": np.average(variance_entropy_clean),
            "weighted_sparsity": np.average(a=sparsity, weights=weights),
            "weighted_variance_entropy": np.average(a=variance_entropy, weights=weights),
            "weighted_variance_entropy_clean": np.average(a=variance_entropy_clean, weights=weights),
        }


class ArchitectureSizeMetrics(Metrics):
    """
    Architecture Size as metrics. The sizes of the different layer are also calculated
    """

    def __init__(self):
        super().__init__()

    def calculate_metrics(self, model) -> dict:
        """
        Calculates architecture size metrics from given model
        :param model: model to calculate metrics from
        :return: dict of calculated architecture size metrics
        """
        children = get_children(model)

        names = list(dict.fromkeys([item.__class__.__name__ for item in children]))

        architecture_size_dict = {
            f"architecture_size_{name}": len([item for item in children if str(item.__class__.__name__) == name]) for
            name in names}
        architecture_size_dict['architecture_size'] = len(children)

        return architecture_size_dict


class InformationalMetrics(Metrics):

    def __init__(self):
        super().__init__()

    def calculate_metrics(self, model) -> dict:
        return {
            "dataset": model.myhparams["dataset"]
        }


class LatentSpaceMetrics(Metrics):
    """
    Metric based on the silhouette score of the latent space. The latent space comes before the fc-Layer and
    contains all the feature maps, which can be clustered.
    """

    def __init__(self, cluster_model_name, **clustering_model_static_kwargs):
        """
        Initialize the metric calculation.
        :param cluster_model_name: Name of the used cluster_model
        :param clustering_model_static_kwargs: static parameters for the cluster model
        """
        super().__init__()
        assert cluster_model_name in ['kmeans', 'KMeans', 'dbscan', 'DBScan']
        self.cluster_model_name = cluster_model_name
        self.clustering_model_static_kwargs = clustering_model_static_kwargs

    def __get_clustering_model(self, cluster_model_name):
        """
        Gets the cluster model class and the dynamic parameters, which need to initialized while the
        clustering process is running. Like number of cluster etc.
        :param cluster_model_name: name of cluster model
        :return: cluster model class and the dynamic parameters as tuple
        (cluster_model_class, dynamic_parameters as kwargs)
        """
        cluster_model_dynamic_kwargs = None
        cluster_model = None

        if cluster_model_name in ['kmeans', 'KMeans']:
            cluster_model_dynamic_kwargs = {'n_clusters': None}
            cluster_model = KMeans

        if cluster_model_name in ['dbscan', 'DBScan']:
            cluster_model_dynamic_kwargs = {'eps': None}
            if 'min_samples' not in self.clustering_model_static_kwargs:
                cluster_model_dynamic_kwargs['min_samples'] = None
            cluster_model = DBSCAN

        return cluster_model, cluster_model_dynamic_kwargs

    def __get_dynamic_params(self, latent_space, model, cluster_model_dynamic_kwargs):
        """
        Calculates the dynamic parameters given
        :param latent_space: latent space, which to cluster
        :param model: dynamic model
        :param cluster_model_dynamic_kwargs: dynamic parameters, which to be initialized
        :return: initialized dynamic parameters as dict
        """
        result = {}
        if cluster_model_dynamic_kwargs is not None:
            result.update(cluster_model_dynamic_kwargs)

        if cluster_model_dynamic_kwargs is not None and 'n_clusters' in cluster_model_dynamic_kwargs and \
                cluster_model_dynamic_kwargs['n_clusters'] is None:
            result['n_clusters'] = int(model.myhparams['num_classes'])

        if cluster_model_dynamic_kwargs is not None and 'eps' in cluster_model_dynamic_kwargs and \
                cluster_model_dynamic_kwargs['eps'] is None:

            if self.clustering_model_static_kwargs is not None and 'min_samples' in self.clustering_model_static_kwargs:
                min_samples = self.clustering_model_static_kwargs['min_samples']
            else:
                min_samples = len(np.shape(latent_space)) * 2

            nearest_neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors = nearest_neighbors.fit(latent_space)

            distances, indices = neighbors.kneighbors(latent_space)
            distances = np.sort(distances[:, min_samples - 1], axis=0)

            i = np.arange(len(distances))
            knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

            result['eps'] = distances[knee.knee]

        if cluster_model_dynamic_kwargs is not None and 'min_samples' in cluster_model_dynamic_kwargs and \
                cluster_model_dynamic_kwargs['min_samples'] is None:
            result['min_samples'] = len(np.shape(latent_space)) * 2

        return result

    def calculate_metrics(self, model) -> dict:
        """
        Calculates latent space metrics from given model with a clustering algorithm
        :param model: model to calculate metrics from
        :return: dict of calculated latent space metrics
        """
        if 'finetune_dataset' in self.config:
            data_dir = os.path.join(model.myhparams["data_dir"], self.config['finetune_dataset'][0])
        else:
            data_dir = os.path.join(model.myhparams["data_dir"], model.myhparams["dataset"])
        data_ = None

        if model.myhparams['dataset_percentage'] == 100:
            data_ = data.get_dataset(model.myhparams["dataset"])(data_dir, model.myhparams["batch_size"],
                                                                 model.myhparams["num_workers"])
        if 100 > model.myhparams['dataset_percentage'] > 0:
            data_ = data.get_dataset_minimized(model.myhparams["dataset"])(data_dir,
                                                                           model.myhparams["batch_size"],
                                                                           model.myhparams["num_workers"],
                                                                           model.myhparams[
                                                                               'dataset_percentage'])

        # register forward hook on latent space (bevor classifier)
        hook = LayerHook()
        hook.register_hook(model.model.fc)

        # train a batch
        model(next(iter(data_.train_dataloader())))

        latent_space = hook.pull()[0].detach()

        print(f'{"model":13}: {model.myhparams["classifier"]}')

        print(f'{"shape":13}: {tuple(np.shape(latent_space))}')
        print(f'{"cluster_algo":13}: {self.cluster_model_name}')

        all_params = {}

        clustering_model_class, dynamic_cluster_kwargs = self.__get_clustering_model(self.cluster_model_name)

        all_params.update(self.clustering_model_static_kwargs)
        all_params.update(self.__get_dynamic_params(
            latent_space,
            model,
            dynamic_cluster_kwargs
        ))
        if 'dbscan' == self.cluster_model_name and 'eps' in all_params and 'min_samples' in all_params:
            print(f'{"eps":13}: {all_params["eps"]:2.2f}')
            print(f'{"min_samples":13}: {all_params["min_samples"]}')

        cluster_model = clustering_model_class(**all_params)
        cluster_model.fit_predict(latent_space)

        score = silhouette_score(latent_space, cluster_model.labels_, metric='cosine')

        return {"latent_space_silhouette_score": score}
