# Beyond calibration: estimating the grouping loss of modern neural networks
This repository reproduces the results of the paper ["Beyond calibration: estimating the grouping loss of modern neural networks"](https://arxiv.org/abs/2210.16315) by Alexandre Perez-Lebel, Marine Le Morvan, and Gaël Varoquaux (ICLR 2023).


## User-friendly package for estimating the grouping loss [Coming soon]
A separate package to easily estimate the grouping loss of a classifier will be available at:
https://github.com/aperezlebel/gl_estimation.


## Install

```bash
git clone https://github.com/aperezlebel/beyond_calibration
```

```bash
conda install --file requirements.txt -c conda-forge
```

## Reproduce
`src/test_figures.py` generates the figures present in the paper. Each is written as a test function that can be run with `pytest`.

Example:
```bash
# Generate the first figure of the paper.
pytest src/test_figures.py::test_fig1 -s
```

Depending on the figures you want to reproduce, you may need to install data.
* **Figures 1 to 5 and 9 to 12**: no data required. You can already run the commands.
* **Figures 6 to 8 and 13 to 27**: ⚠️ data required. You should install the data before running the commands: see the section 'Full data build' below.


### Full data build (required for Figures 6 to 8 and 13 to 27)
This procedure builds all the datasets, enabling the reproduction of all the figures (main text + appendix). If you want to reproduce only a subset of the figures, jump to the 'Partial data build for specific figures only' section.

#### 1. Make the datasets
This downloads all the dataset archives (ImageNet-1K validation set, ImageNet-R, and ImageNet-C), extracts them, and builds the merged version of ImageNet-C.
```bash
pytest src/test_data.py::test_make_datasets -s
```


<details>
  <summary>Details.</summary>
  The above is equivalent to running the following commands separately.

  #### 1.1 Download dataset archives
  This downloads the dataset archives of ImageNet-1K (val), ImageNet-R, and ImageNet-C.
  ```bash
  pytest src/test_data.py::test_download_datasets -s
  ```
  #### 1.2. Extract dataset archives
  ```bash
  pytest src/test_data.py::test_extract_datasets -s
  ```
  #### 1.3. Create ImageNet-C merged dataset
  This is a manually created dataset from corruptions of ImageNet-C. More details are in section D.2 of the article.
  ```bash
  pytest src/test_data.py::test_make_imagenet_c_merged_no_rep -s
  ```

</details>

#### 2. Download pre-trained networks
```bash
pytest -n 15 src/test_data.py::test_download_vision_networks -s
pytest -n 2 src/test_data.py::test_download_nlp_network -s
```
#### 3. Forward networks
Since we work in the last layer's feature space, we forward once and for all the datasets through each network, creating as many datasets of embeddings. The evaluation then only looks at those smaller datasets.
```bash
pytest -n 30 src/test_data.py::test_forward_vision_networks -s
pytest -n 2 src/test_data.py::test_forward_nlp_network -s
```



### Partial data build for specific figures only (faster)
Depending on the figures you want to reproduce, build a subset of the data as follows:
| Figure   | Command                                                        |
|----------|----------------------------------------------------------------|
| Figure 6 |  `pytest src/test_data.py::test_fig6_requirement -s --njobs 2`      |
| Figure 7 |  `pytest src/test_data.py::test_fig7_requirement -s --njobs 15`      |
| Figure 8 |  `pytest src/test_data.py::test_fig8_requirement -s --njobs 2`      |

<details>
  <summary>Click for appendix Figures 13 to 27.</summary>

| Figure   | Command                                                        |
|----------|----------------------------------------------------------------|
| Figure 13 |  `pytest src/test_data.py::test_fig13_requirement -s --njobs 15`      |
| Figure 14 |  `pytest src/test_data.py::test_fig14_requirement -s --njobs 30`      |
| Figure 15 |  `pytest src/test_data.py::test_fig15_requirement -s --njobs 15`      |
| Figure 16 |  `pytest src/test_data.py::test_fig16_requirement -s --njobs 15`      |
| Figure 17 |  `pytest src/test_data.py::test_fig17_requirement -s --njobs 15`      |
| Figure 18 |  `pytest src/test_data.py::test_fig18_requirement -s --njobs 15`      |
| Figure 19 |  `pytest src/test_data.py::test_fig19_requirement -s --njobs 15`      |
| Figure 20 |  `pytest src/test_data.py::test_fig20_requirement -s --njobs 15`      |
| Figure 21 |  `pytest src/test_data.py::test_fig21_requirement -s --njobs 15`      |
| Figure 22 |  `pytest src/test_data.py::test_fig22_requirement -s --njobs 15`      |
| Figure 23 |  `pytest src/test_data.py::test_fig23_requirement -s --njobs 15`      |
| Figure 24 |  `pytest src/test_data.py::test_fig24_requirement -s --njobs 15`      |
| Figure 25 |  `pytest src/test_data.py::test_fig25_requirement -s --njobs 15`      |
| Figure 26 |  `pytest src/test_data.py::test_fig26_requirement -s --njobs 15`      |
| Figure 27 |  `pytest src/test_data.py::test_fig27_requirement -s --njobs 15`      |
</details>

### List of figures

| Figure   | Requires <br> data | Resource <br> intensive | Command  |
|----------|---------------|--------------------|---|
| Figure 1 |       <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">      |    <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig1 -s`   |
| Figure 2 |       <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig2 -s`   |
| Figure 3 |       <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig3 -s`   |
| Figure 4 |       <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">            | `pytest src/test_figures.py::test_fig4 -s --njobs 120`   |
| Figure 5 |       <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig5 -s`   |
| Figure 6 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig6 -s -n 4 --njobs 15`   |
| Figure 7 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig7 -s --njobs 15`   |
| Figure 8 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig8 -s -n 4 --njobs 15`   |

<details>
  <summary>Click for appendix Figures 9 to 27.</summary>

| Figure   | Requires <br> data | Resource <br> intensive | Command  |
|----------|---------------|--------------------|---|
| Figure 9 |       <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig9 -s`   |
| Figure 10 |       <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig10 -s`   |
| Figure 11 |       <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig11 -s`   |
| Figure 12 |       <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/No-Green.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig12 -s`   |
| Figure 13 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig13 -s --njobs 15`   |
| Figure 14 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig14 -s --njobs 120`   |
| Figure 15 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig15 -s -n 15 --njobs 8`   |
| Figure 16 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig16 -s -n 15 --njobs 8`   |
| Figure 17 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig17 -s -n 15 --njobs 8`   |
| Figure 18 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig18 -s -n 15 --njobs 8`   |
| Figure 19 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig19 -s -n 15 --njobs 8`   |
| Figure 20 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig20 -s -n 15 --njobs 8`   |
| Figure 21 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig21 -s -n 15 --njobs 8`   |
| Figure 22 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig22 -s -n 15 --njobs 8`   |
| Figure 23 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig23 -s -n 15 --njobs 8`   |
| Figure 24 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig24 -s -n 15 --njobs 8`   |
| Figure 25 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig25 -s -n 15 --njobs 8`   |
| Figure 26 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig26 -s -n 15 --njobs 8`   |
| Figure 27 |       <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">     |    <img src="https://img.shields.io/badge/Yes-yellow.svg?logo=LOGO">             | `pytest src/test_figures.py::test_fig27 -s -n 15 --njobs 8`   |
</details>

**Comments:**
* Figures marked as 'resource intensive' are recommended to be run on a computing cluster. The complete experiments were run on a 256-CPU node for several days. The expensive part is to forward the datasets through the networks to create datasets of embeddings of inputs in the last layer feature space. Then, the evaluation of the grouping loss with the partitioning is fast.
* Some tests are parallelized using the `pytest-xdist` plugin through the `-n` argument or internally using the `--njobs` argument. When specified, adjust the number of workers (`-n` or `--njobs`) depending on your node's CPU count.
* Add `--disable-warnings` to the pytest command to silent warnings.

## Files
* `src/test_data.py`: code building the datasets necessary to reproduce the experiments.
* `src/test_figures.py`: code generating the figures present in the paper.

* `src/partitioning.py`: main partitioning algorithm (implemented in the `cluster_evaluate` function). It partitions the feature space in each level set and returns the bins' region scores, counts, and average confidence scores.
* `src/networks/*`: code related to vision and NLP networks. All networks inherit the BaseNet class in `src/networks/base.py`, which implements functions that load the networks, forward samples, extract transformed samples in the high-level feature space, confidence scores, etc...
* `_utils.py`, `_plot.py`, `_linalg.py` are implementing helper functions.
* `tests/*`: unit tests to test the functions of the repository.


## Contact

Should you have any questions, comments, or feedback, please open an issue or reach out!
* Email: alexandre [dot] perez [at] inria [dot] fr
* Twitter: [@aperezlebel](https://twitter.com/aperezlebel)
* Website: https://perez-lebel.com
