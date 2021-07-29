# nested-effects

    .
    ├── demos   # Demonstration notebooks for a selection of models
    ├── paper   # Materials pertaining to the academic paper
    ├── tests   # Test root
    └── nfx     # Code root


## Instructions for Ubuntu/OSX

0. Install the generic dependencies: `Python 3.8`, `pip`, `pipenv`, `git`. Also install `suite-sparse`. On Ubuntu:

    ```shell
    sudo apt-get install libsuitesparse-dev
    ```

    On OSX:

    ```shell
    brew install suite-sparse
    ```

1. Define your project root (`[project]`) and navigate there

    ```shell
    mkdir [project]
    cd [project]
    ```

2. Clone the repository

    ```shell
    git clone https://github.com/timsf/crossed-effects.git
    ```

3. Create a virtual environment and install dependencies

    ```shell
    pipenv install
    ```

4. Start a shell within the virtual environment

    ```shell
    pipenv shell
    ```

5. Start the `Jupyter` server

    ```shell
    jupyter notebook
    ```

6. Access the `Jupyter` server in your browser and navigate to the notebook of interest
