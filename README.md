# Formalizing the Binding Problem (ICML 2026)

[![Paper](https://img.shields.io/badge/Paper-arXiv-8A2BE2)](https://arxiv.org/abs/2606.03976)

We probe feature-binding information in frozen vision backbones. The training pipeline builds a binding dataset, caches frozen model activations, trains one or more probes, and reports probe loss, accuracy, binding information, and binding ratio.

## Setup

Create or activate a Python environment, then install the Python dependencies:

```bash
pip install -r requirements.txt
```

PyTorch installation may need to be adjusted for your CUDA version. If `pip install torch` does not select the right build, install PyTorch from the official PyTorch instructions first, then install the rest of the requirements.

## Repository Layout

```text
src/
  main.py                         # entry point for all runs
  trainer.py                      # training, binding evaluation
  activation.py                   # obtain model activations
  probes.py                       # probe architectures
  cfgs/
    config.yaml                   # main experiment config
    dataset/                      # dataset configs
  dataset/
    ColorShape.py
    BalancedColorShape.py
    CLEVR.py
    OcclusionClevr.py
    VGColor.py
    VGTopAttr.py
    script/
      occlusionclevr_blender_generation_script.py
```

## Supported Datasets

`balancedcolorshape` dataset is used in Section 3.1, 3.2 of the ppaer. All other datasets including `colorshape`, `occlusionclevr`, `clevr`, `vgcolor`, and `vgtopattr` are used in Section 3.3. The paper including the appendix contains more detailed setups. Each has their specific config file under `src/cfgs/dataset`. Datasets obtained from external sources/generators are detailed below:

### OcclusionClevr

`occlusionclevr` generates or loads CLEVR-style images using Blender and a local clone of `facebookresearch/clevr-dataset-gen` (see this repo for instructions on installing Blender):

```bash
git clone https://github.com/facebookresearch/clevr-dataset-gen.git
export CLEVR_GEN_DIR=/path/to/clevr-dataset-gen
export BLENDER_PATH=/path/to/blender
```

The default location for generated images is:

```text
data/generated/occlusionclevr
```

### CLEVR

The `clevr` loader expects a derived 6-object CLEVR layout:

```text
${CLEVR_DATA_DIR}/
  images_6obj/
    CLEVR_6obj_000000.png
  scenes_6obj/
    CLEVR_6obj_000000.json
```

Set:

```bash
export CLEVR_DATA_DIR=/path/to/derived/clevr
```

### Visual Genome

The `vgcolor` and `vgtopattr` loaders expect Visual Genome images plus project-specific mined metadata:

```text
${VG_DATA_DIR}/
  VG_100K/
  VG_100K_2/
  deprecated/
    meta_filtered_color_coco/
      attributes_filtered_color.json
      vg_stats_filtered_color.json
    meta_filtered_topattr_coco/
      attributes_filtered_topattr.json
      vg_stats_filtered_topattr.json
```

Set:

```bash
export VG_DATA_DIR=/path/to/visual_genome
export VG_IMG_ROOT=/path/to/visual_genome
```


## Models, Activations, and Probes

Backbones, activations, and probes are configured in `src/cfgs/config.yaml`.

Supported activation modes:

| Mode | Meaning |
|---|---|
| `cls` | The CLS token. |
| `mean_spatial` | The mean-pooled representation of all spatial tokens. |
| `all` | All spatial token representations. |
| `cls_mean` | The concatenation of the CLS token and the mean-pooled spatial tokens. |

Supported probe types:

| Probe type | Brief explanation | Required setup (`config.yaml`) |
|---|---|---|
| `linear` | Linear readout. | `use_conditioned_query` |
| `dnn_concat` / `dnn` | MLP. | `dnn_hidden_dim`; `dnn_num_layers`; `dnn_dropout`; `use_conditioned_query` |
| `quadratic_concat` | Low-rank quadratic readout. | `rank`; `use_conditioned_query` |
| `quadratic_concat_reuse` | Quadratic probe with shared parameters for same feature type. Unavailable as feature probes. | `rank`; `use_conditioned_query` |
| `multilinear_concat_reuse` | Generalized reused probe for multiple feature types. Unavailable as feature probes. |  `rank`; `feature_dims`; `use_conditioned_query` |
| `attention_quadratic` | Attention over tokens, followed by a quadratic readout. | `rank`; `use_conditioned_query` |

Attention probes require all spatial tokens, while all other probes can take different activation modes:

```yaml
model:
  output_mode: all # required for attention
```

## Running Experiments

Adjust dataset specific settings and experiment settings (dataset type, probe type) in their corresponding `.yaml` config files (see above).

Run from the repository root:

```bash
python src/main.py
```

Outputs are written under:

```text
data/outputs/${dataset.name}/...
```

Frozen backbone activations are cached under:

```text
data/cache/activations/
```

**Happy binding!**