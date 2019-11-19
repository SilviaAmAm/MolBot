# Generative Recurrent Neural Networks


This package implements the Generative Recurrent Neural Networks from the [paper](https://onlinelibrary.wiley.com/doi/10.1002/minf.201700111)  from Gupta et al. and combines it with the Reinforcement Learning procedures described in the [paper](https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-017-0235-x) from Olivecrona et al.

## Current List of Contributors:


- Silvia Amabilino (NovaData Solutions ltd., University of Bristol)
- Michael Mazanetz (NovaData Solutions ltd.)
- David Glowacki (University of Bristol)

## Installation


In order to use this package, you need the following packages installed:

- scikit-learn (0.19.1 or higher)
- tensorflow (1.9.0 or higher, but < tensorflow 2.0) or tensorflow-gpu (1.9.0 or higher, but < tensorflow 2.0)
- keras (2.2.0 or higher)
- rdkit (optional)

RDkit is only needed for the reward function in reinforcement learning.

To install, you can run the following command:

```
pip install git+https://github.com/SilviaAmAm/MolBot.git
```

Otherwise, you can clone the repository:

```
git clone https://github.com/SilviaAmAm/MolBot.git
```

and then, in your desired python environment, run:

```
cd MolBot
pip install MolBot/
```

## Building the documentation

To build the documentation, these packages are required:

- nbsphinx
- sphinx_rtd_theme

In the [docs](./docs) folder, run:

```
make html
```

Then, open `./docs/build/html/index.html`.

## Usage

You can have a look at some examples in the [examples](./examples) folder.

## Running the tests


Go to the test directory:

``cd tests``

Then, run all the tests:

``pytest test_*``


