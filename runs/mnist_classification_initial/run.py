
from jax.typing import ArrayLike

from cweb.model import *
from cweb.states import *
from cweb.utils import *
from cweb.helpers import random_phases
from cweb.sample import apply_unitary_and_measure

import math
import jax
import jax.numpy as jnp
import pandas as pd
import os
import pathlib

LABEL = 'label'

def initialise_model(
        key : ArrayLike, 
        m : int, 
        n : int, 
        data_dim : int, 
        n_classes : int
    ) -> tuple[ReuploadingClements,AffineLayer]:
    """
    Initialises the model using a random key.

    Args:
        key (ArrayLike) : The random key used to generate parameters.
        m (int)         : The number of photonice modes.
        n (int)         : The number of photons input.
        data_dim (int)  : The dimension of the input data.
        n_classes (int) : The number of output classes.

    Returns:
        tuple[ReuploadingClements,AffineLayer] : A pytree (in this can tuple of dataclasses) representing the model.
    """

    # The first column reuploads the remaining are  parameterised
    columns = [ReuploadingClementsColumn(data_dim, jnp.arange(m), is_primary=True)]
    for i in range(1, m):
        key, subkey = jax.random.split(key)
        is_primary = (i % 2) == 0
        columns.append(ClementsColumn(random_phases(subkey, m), is_primary=is_primary))

    key, subkey = jax.random.split(key)
    rc = ReuploadingClements(
        diagonal_phases=random_phases(jax.random.fold_in(subkey, 0), m),
        columns=columns,
        data_dim=data_dim
    )

    # Generate affine transformation for final layer
    n_fock_states = math.comb(n + m - 1, n)

    key, subkey = jax.random.split(key)
    weights = jax.random.normal(subkey, (n_classes, n_fock_states))

    key, subkey = jax.random.split(key)
    biases = jax.random.normal(subkey, n_classes)

    al = AffineLayer(
        weights=weights,
        biases=biases
    )

    return (rc, al)

def load_mnist_pca_data(is_train : bool) -> tuple[jax.Array, jax.Array]:
    """
    Loads the MNIST PCA data (as prepared in notebook repo) from either the train or test split.

    Args:
        is_train (bool) : Whether the load the train or test data.

    Returns:
        tuple[jax.Array, jax.Array] : The loaded data as JAX arrays (pca data, labels).
    """

    # Select the csv file
    csv_path = f'./data/mnist_pca/{"train" if is_train else "test"}_mnist_pca.csv'

    # Read the csv file
    df = pd.read_csv(csv_path, index_col=0)

    # Extract and convert data
    labels   = jnp.array(df[LABEL].to_numpy())
    pca_data = jnp.array(df.drop(columns=[LABEL]).to_numpy())

    return pca_data, labels

def execute_run(key : ArrayLike, logger : Logger, data_logger : DataLogger):
    m = 8
    n = 3
    data_dim = 32
    n_classes = 10
    batch_size = 128

    # Initialise the model
    logger.log('Initialising model...')
    key, subkey = jax.random.split(key)
    initial_model = initialise_model(
        key=subkey,
        m=m,
        n=n,
        data_dim=data_dim,
        n_classes=n_classes)
    logger.log('Done.')

    # Load data
    logger.log('Loading MNIST PCA data...')
    train_data = load_mnist_pca_data(is_train=True)
    test_data  = load_mnist_pca_data(is_train=False)

    train_pca_data, _ = train_data
    test_pca_data,  _ = test_data

    logger.log(f'Train data shape: {train_pca_data.shape}.')
    logger.log(f'Test  data shape: {test_pca_data.shape }.')

    assert train_pca_data.shape == (60_000, 32)
    assert  test_pca_data.shape == (10_000, 32)
    logger.log('Done.')

    # Photon state used
    photon_input_state = FixedPhotonCountMixedFockState(
        fock_states=jnp.array([[0, 1, 2]]),
        probabilities=jnp.array([1.0])
    )
    logger.log(f'Initial photon input state: {photon_input_state}.')


    # Define the loss function
    @jax.jit
    def loss_fn(
            model : tuple[ReuploadingClements, AffineLayer], 
            data : tuple[jax.Array, jax.Array]
        ) -> jax.Array:
        """
        The loss function for the model parameters is calculated by first transforming the input photon state through a parameterised reuploading clements.
        The output probability distribution this gives is then transformed via the affine layer and a softmax is applied to give the final q distribution.
        The loss itself is the cross entropy of this against a perfect classifier matching a one-hot encoding.

        Args:
            model (tuple[ReuploadingClements, AffineLayer]) : The model as a tuple of the two parts (a reuploading clements and an affine layer).
            data (tuple[jax.Array, jax.Array])              : The data as two matricers: data (N,32) and labels (N,1).

        Returns:
            jax.Array : The mean loss overa all N examples.
        """

        rc , al = model
        pca_data, labels = data

        # With PCA data generate unitaries via clements scheme
        c : Clements = rc.get_clements(data=pca_data)
        u = c.get_unitary()

        # Generate output mixed state
        output_state = apply_unitary_and_measure(u, photon_input_state)

        # Transform output_state probabilities
        q = softmax(al(output_state.probabilities))

        # Callculate loss
        one_hot_labels = jnp.arange(n_classes)[None,:] == labels[:,None]
        p = one_hot_labels

        loss = jnp.mean(cross_entropy(p, q))

        return loss

    # Check the test loss
    initial_test_loss = jax.block_until_ready(loss_fn(initial_model,  test_data))
    logger.log(f'Initial test loss: {initial_test_loss:.3e}')



def main():
    # Setup output folder
    output_folder = str(
        pathlib.Path.home()
        /'work'
        /'cweb'
        /'mnist_classification_initial'
    )

    # Initialise logger
    logger = Logger(output_folder)
    logger.log('Running "MNIST classification initial"...')

    # Execute the run
    data_logger = DataLogger(output_folder)
    execute_run(
        key=jax.random.key(0),
        logger=logger,
        data_logger=data_logger)

    logger.log('Finishes!')

if __name__ == '__main__':
    main()