# Cometh : A Continuous-Time Discrete-State Graph Diffusion Model

Official Pytorch implementation of Cometh : A Continuous-Time Discrete-State Graph Diffusion Model.
Paper link : [https://arxiv.org/abs/2406.06449](https://arxiv.org/abs/2406.06449)

## 🧱 Environment Installation

We use **Conda** to manage the environment. All dependencies are specified in the provided configuration file [`cometh.yml`](./cometh.yml).

### 🔧 Step 1: Create the Conda Environment

Run the following command from the root directory of the repository:

```bash
conda env create -f cometh.yml
conda activate cometh
```
### 🔧 Step 2: Compile orca 

The evaluation on synthetic graphs requires to compile orca. Navigate to the ./src/analysis/orca directory and compile orca.cpp:

```bash
cd ./src/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp
```


## 🚀 Run the code

- To train the model, e.g. on QM9, run ```python main.py +experiment=qm9.yaml encoding=rrwp```
- Since we use different hyperparameters for RRWP depending on the dataset, the encoding config differs from one dataset to another. The corresponding argument for 'encoding' are rrwp for QM9, rrwp_planar for Planar and rrwp_moses for SBM, MOSES and GuacaMol.

## 📍 Checkpoints and Inference

We provide checkpoints for all the datasets in this [folder](https://drive.google.com/drive/folders/1bRct8zRDpYb_WkY4adtjWWuJi3WpwZ0v?usp=sharing)

For each dataset, there are two checkoints, one for the original model and one the EMA weights. To load a model and sample from it, place the two checkpoints in the same folder and run : 

```python main.py +experiment=qm9_sampling.yaml encoding=rrwp general.test_only="path_to_your_checkpoint"```

## 📚 Citation
```bibtex
@article{siraudin2024cometh,
  title     = {Cometh: A continuous-time discrete-state graph diffusion model},
  author    = {Antoine Siraudin and Fragkiskos D. Malliaros and Christopher Morris},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.06449}
}
```
