import os

from torchconvquality import measure_quality
from sklearn.metrics import silhouette_score
import numpy as np
from metrics.utils import get_children

from metrics.loader.layer_hook import LayerHook
import data


class Metrics:
    def __init__(self):
        pass

    def calculate_metrics(self, model) -> dict:
        return {}


class QualityMetrics(Metrics):
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
    def __init__(self, clustering_model, **clustering_model_kwargs):
        super().__init__()
        self.clustering_model = clustering_model(clustering_model_kwargs)

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

        # if len(np.shape(latent_space)) == 4:
        print(model.myhparams['classifier'])
        print(np.shape(latent_space))

        # if len(np.shape(latent_space)) == 4:
        #     latent_space=latent_space.reshape(np.shape(latent_space)[0] * np.shape(latent_space)[2], np.shape(latent_space)[1] * np.shape(latent_space)[3])

        # calculate silhouette score with kmeans
        # KMeans(n_clusters=model.myhparams['num_classes'], random_state=42)

        self.clustering_model.fit_predict(latent_space)

        score = silhouette_score(latent_space, self.clustering_model.labels_, metric='cosine')

        return {"latent_space_silhouette_score": score}
