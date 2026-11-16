<div align="center">

# Fast or Slow? Predict AI Model Runtime

### Ranking compiler configurations with graph transformers

[![Kaggle](https://img.shields.io/badge/Kaggle-10th%20Place-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/predict-ai-model-runtime)
[![NeurIPS](https://img.shields.io/badge/NeurIPS%202023-TpuGraphs-8C1515?style=for-the-badge)](https://arxiv.org/abs/2308.13490)

**Intermediate fusion + linear attention + ListMLE for the layout collections of TpuGraphs**

</div>

---

## Problem

[TpuGraphs](https://github.com/google-research-datasets/tpu_graphs) is a dataset of XLA computational graphs, compiler configurations and runtimes measured on TPUs. The task of the [Kaggle competition](https://www.kaggle.com/competitions/predict-ai-model-runtime) is to rank the configurations of unseen graphs by runtime. Layout collections are scored with Kendall's tau, the tile collection with the slowdown of the top 5 predictions.

Our team (DrHB and [Iafoss](https://www.kaggle.com/iafoss)) finished 10th of 616. The full description is in the [solution write-up](https://www.kaggle.com/competitions/predict-ai-model-runtime/writeups/di-10th-place-solution-fast-or-slow-with-graph-tra).

## What is in this repository

This is my working code for the **layout** model as of 3 November 2023, two weeks before the deadline. It is not the complete team solution: the tile model, the training scripts, the APPNP and DIFFormer variants and the ensembling are not here.

| file | contents |
| --- | --- |
| `eda/convert_data1.ipynb` | Converts the layout `.npz` files. Graphs go into one pickle that stays in memory. The configurations of every graph are cut into files of 1,000, so a training step reads one small file. |
| `tpuoptimize/dataset.py` | `TPUDatasetV0`. One item is one graph with a random sample of its configurations. Node features go through `log(x + 3)`. |
| `tpuoptimize/models.py` | `MSLA`, multi-head linear attention with an ELU + 1 kernel. `GraphResABlock`, a transformer block with linear attention and a SAGE layer in place of the MLP. `TPUModelV0`, the full model. |
| `tpuoptimize/utils.py` | ListMLE loss, a pairwise ranking loss, and Kendall's tau metrics per collection. |
| `tpuoptimize/fastai_fit.py` | Mixed precision callback for fastai. |
| `nbs/` | The nbdev notebooks the modules are exported from. |

## Technical highlights

- **Intermediate fusion.** The graph encoder runs once per graph. The configuration is joined to the embeddings of the configurable nodes only, after the encoder and before pooling. The cost of the encoder does not depend on the number of configurations, so about 1,000 configurations of a graph with 10k nodes fit in one step.
- **Linear attention.** Attention with linear cost in the number of nodes gives a global receptive field over the whole graph on a single GPU.
- **Graph layer in place of the MLP.** Each block is a global step (attention) followed by a local one (message passing along the edges).
- **ListMLE.** A listwise ranking loss over all sampled configurations of a graph.

## Model

```
node features (L, 140) -> log(x + 3) -> linear  \
opcode (L,)            -> embedding              +-> (L, 256)
is configurable (L,)   -> embedding             /

12 x [ x + LinearAttention(LN(x)) ;  x + SAGE(GELU(LN(x))) ]

take the L' configurable nodes              (L', 256)
join the embedded layout of each config     (C, L', 256 + 72)
MLP per node                                (C, L', 256)
mean + GeM pooling over nodes               (C, 512)
MLP                                         (C,)   one score per configuration
```

## Usage

Download the data from the [competition page](https://www.kaggle.com/competitions/predict-ai-model-runtime/data), run `eda/convert_data1.ipynb` to convert the layout collections, then:

```python
import glob, os, torch
from tpuoptimize.dataset import TPUDatasetV0
from tpuoptimize.models import TPUModelV0
from tpuoptimize.utils import listMLE_loss

path = 'data/layout_converted_z2_1000'
files = glob.glob(os.path.join(path, 'xla', 'random', 'train', '**/*.npz'), recursive=True)
ds = TPUDatasetV0(files, path=path, nmax=256)          # 256 configurations per step
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

model = TPUModelV0(nh=256, depth=12, psg=0.1, ps=0.1)
x, y = next(iter(dl))
loss = listMLE_loss(model(x), y)
```

Requirements: `torch`, `torch_geometric`, `fastai`, `scipy`, `numpy`.

## Results

| submission | public | private |
| --- | --- | --- |
| Blend of 10 to 20 configurations per model family | 0.712 | 0.696 |
| 2 APPNP models (XLA) + 2 DIFFormer models (NLP), final | 0.721 | 0.703 |
| APPNP with gated units, not selected | 0.706 | 0.715 |

## Dataset collections

| Subtype            | Optimization | Source | Search  | Description | Total graphs + configs |
| ------------------ | ------------ | ------ | ------- | ----------- | ---------------------- |
| Layout:XLA:Default | Layout | XLA | Default | Layouts from a genetic algorithm started at the default configuration. Graphs from XLA benchmarks covering various ML models. | 771,496 |
| Layout:XLA:Random  | Layout | XLA | Random  | Layouts from random search. Graphs from XLA benchmarks covering various ML models. | 908,561 |
| Layout:NLP:Default | Layout | NLP | Default | Layouts from a genetic algorithm started at the default configuration. Graphs from BERT models. | 13,285,415 |
| Layout:NLP:Random  | Layout | NLP | Random  | Layouts from random search. Graphs from BERT models. | 16,125,781 |
| Tile:XLA           | Tile   | XLA | N/A     | All tile sizes are enumerated, as the search space is small. Graphs from XLA benchmarks covering various ML models. | 12,870,077 |

## References

- Phothilimthana et al. [TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs](https://arxiv.org/abs/2308.13490). NeurIPS 2023, Datasets and Benchmarks.
- Cai et al. [EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction](https://arxiv.org/abs/2205.14756). ICCV 2023.
- Wu et al. [DIFFormer: Scalable (Graph) Transformers Induced by Energy Constrained Diffusion](https://arxiv.org/abs/2301.09474). ICLR 2023.
- Xia et al. Listwise Approach to Learning to Rank: Theory and Algorithm. ICML 2008.
