import os

from sklearn.cluster import KMeans, DBSCAN
from torchconvquality import measure_quality
from sklearn.metrics import silhouette_score
import numpy as np
from metrics.utils import get_children

from metrics.loader.layer_hook import LayerHook
import data


class Metrics:
    """
    Superclass to implement different metrics for analysis of pretrained neural networks
    """

    def __init__(self):
        pass

    def calculate_metrics(self, model) -> dict:
        """
        Abstract method to define required parameters and return values
        :param model: model to calculate metrics from
        :return: dict of calculated metrics
        """
        return {}


class QualityMetrics(Metrics):
    """
    Quality metrics like sparsity and entropy variance. The class also calculates
    the weighted sparsity, entropy variance and the cleaned entropy variance
    """

    def __init__(self, sparsity_eps=0.1):
        super().__init__()
        self.sparsity_eps = sparsity_eps

    def calculate_metrics(self, model) -> dict:
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
        children = get_children(model)

        names = list(dict.fromkeys([item.__class__.__name__ for item in children]))

        architecture_size_dict = {
            f"architecture_size_{name}": len([item for item in children if str(item.__class__.__name__) == name]) for
            name in names}
        architecture_size_dict['architecture_size'] = len(children)

        return architecture_size_dict


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

    @staticmethod
    def __get_clustering_model(cluster_model_name):
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
            cluster_model_dynamic_kwargs = None
            cluster_model = DBSCAN

        return cluster_model, cluster_model_dynamic_kwargs

    @staticmethod
    def __get_dynamic_params(model, cluster_model_dynamic_kwargs):
        """
        Calculates the dynamic parameters given
        :param model: dynamic model
        :param cluster_model_dynamic_kwargs: dynamic parameters, which to be initialized
        :return: initialized dynamic parameters as dict
        """
        result = {}
        if 'n_clusters' in cluster_model_dynamic_kwargs and cluster_model_dynamic_kwargs['n_clusters'] is None:
            result.update(cluster_model_dynamic_kwargs)
            result['n_clusters'] = int(model.myhparams['num_classes'])

        return result

    def calculate_metrics(self, model) -> dict:
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

        print(model.myhparams['classifier'])
        print(np.shape(latent_space))

        all_params = {}

        clustering_model_class, dynamic_cluster_kwargs = self.__get_clustering_model(self.cluster_model_name)

        all_params.update(self.clustering_model_static_kwargs)
        all_params.update(self.__get_dynamic_params(model, dynamic_cluster_kwargs))

        cluster_model = clustering_model_class(**all_params)
        cluster_model.fit_predict(latent_space)

        score = silhouette_score(latent_space, cluster_model.labels_, metric='cosine')

        return {"latent_space_silhouette_score": score}
