# Residual View for DCASE2026 Task 2

Minimal reproduction code for the DCASE2026 Task 2 submitted systems.

Paper link will be added later. This repository keeps only the runnable code
and the main settings used for the reported development-set scores.

## Setup

Install the minimal dependencies:

```bash
python -m pip install -r requirements-minimal.txt
```

Prepare the following external files locally:

- DCASE2026 Task 2 dataset
- pretrained audio encoder checkpoints
- upstream encoder code/checkpoints required by BEATs, SSLAM, and DaSheng

Set dataset paths in:

```text
config/data_config_2026.yaml
```

The example scripts assume these local checkpoint roots:

```text
./transformer-ssl-asd/sslam
./transformer-ssl-asd/beats
```

DaSheng uses the package default checkpoint.

## Run

Generate fixed-10s pair caches and residual-view scores for the final encoder
set:

```bash
bash scripts/run_residual_view_eval_scores.sh
```

Build the PRPS submitted Task 2 folders from the generated pair caches:

```bash
python scripts/build_task2_final_from_pair_cache.py \
  --output_dir out/task2_reproduced
```

Validate a reproduced package:

```bash
python submissions/task2/scripts/validate_submission_package.py \
  --task2_root out/task2_reproduced \
  --systems Kim_LUDO_task2_1,Kim_LUDO_task2_2
```

The pair-cache builder reproduces the PRPS systems, `Kim_LUDO_task2_1` and
`Kim_LUDO_task2_2`. The residual-only SSLAM anchor system, `Kim_LUDO_task2_3`,
is reported as the conservative comparison system.

## Submitted Systems

| system | encoder | PRPS | feature layers |
|:--|:--|:--:|:--|
| `Kim_LUDO_task2_1` | SSLAM | yes | last layer |
| `Kim_LUDO_task2_2` | BEATs iter3, DaSheng-base, SSLAM | yes | BEATs 6/12, DaSheng 1, SSLAM 6/12 |
| `Kim_LUDO_task2_3` | SSLAM | no | last layer |

## Main Settings

| setting | value |
|:--|:--|
| audio crop | fixed 10 s |
| residual coefficient | `alpha = 0.5` |
| MemMix coefficient | `lambda = 0.9` |
| MemMix support | `990` |
| kNN | `K = 1` |
| PRPS prototypes | `128` |
| decision threshold | train-normal 95 percentile |
| domain normalization | off |

## DCASE2026 Development Results

Scores are percentages. The development labels are used only for reporting.

| system | AUC source | AUC target | pAUC | Official Score |
|:--|--:|--:|--:|--:|
| DCASE baseline MAHALA | 66.46 | 54.24 | 53.91 | 57.66 |
| `Kim_LUDO_task2_1` | 70.34 | 68.14 | 57.80 | 63.24 |
| `Kim_LUDO_task2_2` | 71.00 | 67.23 | 55.38 | 62.55 |
| `Kim_LUDO_task2_3` | 67.21 | 64.40 | 56.61 | 62.41 |

Machine-wise official scores:

| machine | baseline MAHALA | System 1 | System 2 | System 3 |
|:--|--:|--:|--:|--:|
| ToyCarEmu | 62.37 | 60.81 | 61.94 | 60.75 |
| ToyCar | 61.33 | 73.33 | 72.98 | 72.77 |
| bearingEmu | 62.79 | 63.07 | 62.57 | 61.15 |
| fan | 51.75 | 50.15 | 51.58 | 50.10 |
| gearboxEmu | 58.92 | 58.36 | 59.44 | 58.55 |
| sliderEmu | 54.29 | 66.90 | 65.46 | 64.54 |
| valveEmu | 54.26 | 78.56 | 68.49 | 76.49 |
| All | 57.66 | 63.24 | 62.55 | 62.41 |

## Not Included

This repository does not redistribute:

- DCASE datasets
- pretrained encoder checkpoints
- generated feature caches
- third-party encoder source trees

Those artifacts must be obtained from their original providers and retain their
own licenses.

## License

The code in this repository is released under the MIT License. External
pretrained encoders, datasets, and upstream repositories are not vendored here
and remain under their own licenses.

## Acknowledgement

This repository is a small modification on top of the GenRep-style frozen
embedding and memory-bank pipeline. We thank the GenRep authors for the
framework and implementation direction. Please also check and cite the original
GenRep work:

```text
https://github.com/Phuriches/GenRepASD
```

```bibtex
@inproceedings{saengthong2025deep,
  title={Deep generic representations for domain-generalized anomalous sound detection},
  author={Saengthong, Phurich and Shinozaki, Takahiro},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}

@techreport{saengthong2025genrep,
  author      = {Saengthong, Phurich and Shinozaki, Takahiro},
  title       = {{GENREP} for first-shot unsupervised anomalous sound detection},
  institution = {DCASE Challenge 2025 Technical Report},
  year        = {2025}
}
```
