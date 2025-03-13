# qnmfinder
A reverse search algorithm for finding QNMs in the ringdown of NR waveforms that utilizes variable projection for free frequency fits and leverages stable complex amplitudes for constructing a physical ringdown model.

This algorithm was developed for the work presented in [arXiv:XXXX]().

A rough schematic of the algorithm can be seen below: 

<img width="922" alt="Workflow" src="https://github.com/user-attachments/assets/b9d562ac-8111-429d-91dc-04bc77d427e1" />

$t_{0}$ is the fitting start time initialized to be some $t_{0}^{f}$, with $t_{0}^{i}$ the earliest fitting start time;
$N_{\mathrm{free}}$ is the number of free frequencies used in the `VarPro` fit initialized to be one; $\Delta t_{0}$ is some amount
by which $t_{0}$ is changed after each failed iteration. For more, see [`build_model`](https://github.com/keefemitman/qnmfinder/blob/65c2319b96bba5cc69a6eb872e905614ad170ae7/qnmfinder/model.py#L918).

## Installation

After cloning, this package can be installed with

```shell
pip install .
```

## Examples and tips

Check out the example in [docs/examples/](https://github.com/keefemitman/qnmfinder/blob/main/docs/examples/example.ipynb)!
