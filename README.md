# nested-effects

Reference implementation for scalable Bayesian crossed effects models.
Algorithms are documented in the paper at https://arxiv.org/abs/2103.10875.

    .
    ├── demos   # Demonstration notebooks for a selection of models
    ├── paper   # Materials pertaining to the academic paper
    ├── tests   # Test root
    └── nfx     # Code root


## Instructions for Ubuntu/OSX

0. Install the generic dependencies: `Python 3.11`, `uv`, `git`. Also install `suite-sparse`. On Ubuntu:

    ```shell
    sudo apt-get install libsuitesparse-dev
    ```

    On OSX:

    ```shell
    brew install suite-sparse
    ```

1. Define your project root (`[project]`) and navigate there:

    ```shell
    mkdir [project]
    cd [project]
    ```

2. Clone the repository:

    ```shell
    git clone https://github.com/timsf/nested-effects.git
    ```

3. Start the `Jupyter` server:

    ```shell
    uv run jupyter notebook
    ```

4. Access the `Jupyter` server in your browser and navigate to the notebook of interest.


## Reference

    @article{Papaspiliopoulos_2023,
        title={Scalable Bayesian computation for crossed and nested hierarchical models},
        volume={17},
        ISSN={1935-7524},
        url={http://dx.doi.org/10.1214/23-EJS2172},
        DOI={10.1214/23-ejs2172},
        number={2},
        journal={Electronic Journal of Statistics},
        publisher={Institute of Mathematical Statistics},
        author={Papaspiliopoulos, Omiros and Stumpf-Fétizon, Timothée and Zanella, Giacomo},
        year={2023},
        month=jan }