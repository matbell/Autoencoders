from clustering_utils import eval_clustering
from datasets import load_subset_context
from autoencoder import Autoencoder
from models.vae import Vae
from utils import log
from sklearn.decomposition import PCA


def test_simple_autoencoder(x, y, pca_dim=None):

    log_file = "simple_ae_results.txt"

    for i in range(2, 50):

        encoding = [(i, 'relu')]
        decoding = [(x.shape[1], 'sigmoid')]

        da = Autoencoder(input_size=x.shape[1], encoding_layers=encoding, decoding_layers=decoding)
        da.fit_model(x)

        encoded_data = da.encoder.predict(x)

        # Performs AE + PCA
        if pca_dim is not None:
            log_file = "simple_ae_pca_results.txt"
            encoded_data = PCA(n_components=pca_dim).fit_transform(encoded_data)

        # Clustering latent points
        k_acc, k_homo, k_n_clusters = eval_clustering(encoded_data, y)
        data = [str(i), str(k_acc), str(k_homo), str(k_n_clusters)]

        print("AE " + str(x.shape[1]) + "-" + str(i) + ": " + str(k_acc))

        # Log clustering results
        log(log_file, ",".join(data))


def test_deep_autoencoder(x, y, pca_dim=None):

    log_file = "deep_ae_results.txt"

    encoding = [(64, 'relu'), (32, 'relu'), (16, 'relu')]
    decoding = [(16, 'relu'), (32, 'relu'), (64, 'relu'), (x.shape[1], 'sigmoid')]

    for i in range(2, 16):

        encoding.append((i, 'relu'))

        da = Autoencoder(input_size=x.shape[1], encoding_layers=encoding, decoding_layers=decoding)
        da.fit_model(x)

        encoded_data = da.encoder.predict(x)

        # Performs AE + PCA
        if pca_dim is not None:
            log_file = "deep_ae_pca_results.txt"
            encoded_data = PCA(n_components=pca_dim).fit_transform(encoded_data)

        # Clustering latent points
        k_acc, k_homo, k_n_clusters = eval_clustering(encoded_data, y)
        data = [str(i), str(k_acc), str(k_homo), str(k_n_clusters)]

        print("Deep AE 64-32-16-" + str(i) + ": " + str(k_acc))

        # Log clustering results
        log(log_file, ",".join(data))


def test_vae(x, y, pca_dim=None):

    log_file = "vae_results.txt"

    intermediate_dims = [125, 64, 32]

    for id in intermediate_dims:

        for ld in range(2, 20):

            vae = Vae(input_dim=x.shape[1], latent_dim=ld, intermediate_dim=id)
            vae.fit_model(x)

            encoded_data = vae.encode(x)

            # Performs AE + PCA
            if pca_dim is not None:
                log_file = "vae_pca_results.txt"
                encoded_data = PCA(n_components=pca_dim).fit_transform(encoded_data)

            # Clustering latent points
            k_acc, k_homo, k_n_clusters = eval_clustering(encoded_data, y)
            data = [str(id), str(ld), str(k_acc), str(k_homo), str(k_n_clusters)]

            print("VAE " + str(id) + "-" + str(ld) + ": " + str(k_acc))

            # Log clustering results
            log(log_file, ",".join(data))


def test_pca(x, y):

    log_file = "pca_results.txt"

    for components in range(2, 20):

        encoded_data = PCA(n_components=components).fit_transform(x)

        # Clustering latent points
        k_acc, k_homo, k_n_clusters = eval_clustering(encoded_data, y)
        data = [str(components), str(k_acc), str(k_homo), str(k_n_clusters)]

        # Log clustering results
        log(log_file, ",".join(data))


def test_clustering(x, y):

    k_acc, k_homo, k_n_clusters = eval_clustering(x, y)
    data = [str(k_acc), str(k_homo), str(k_n_clusters)]

    # Log clustering results
    log("raw_clustering_results.txt", ",".join(data))


if __name__ == '__main__':
    x, y, labels = load_subset_context(data_path='/home/mattia/Development/Sensing/data_analysis/contextLabeler/output',
                                       activities_to_keep=None)

    test_clustering(x, y)
    test_pca(x, y)
    test_simple_autoencoder(x, y)
    test_deep_autoencoder(x, y)
    test_vae(x, y)
