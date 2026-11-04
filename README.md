# Samsung30T-SOP-Estimation

This repository holds the code behind the manuscript on an offline benchmark of sequential state-of-power (SOP) requests with online scale adaptation. The study uses three Samsung INR21700-30T cells from the RPCWBY dataset. It covers:

- the five local predictors and the online scale adaptation;
- the scoring of the benchmark, together with the information-flow and leakage audits and the post hoc analyses;
- the comparators;
- the microcontroller port;
- the scripts that build the tables and figures.

Data and generated results are not included.

## Data

The scripts read the RPCWBY dataset (Borealis, doi:10.5683/SP3/RPCWBY, CC BY 4.0). Put its files in `raw/RPCWBY/` at the repository root:

```
raw/RPCWBY/0_SOP_summary.xlsx
raw/RPCWBY/1_Test_1.zip
raw/RPCWBY/2_Test_2.zip
raw/RPCWBY/3_Test_3.zip
raw/RPCWBY/Arbin Cycler Protocols.zip
```

All paths in the code are relative. Each script finds its inputs from its own location, and writes its outputs into the experiment folder next to it.

Some scripts also read intermediate tables from earlier processing steps that are not part of this repository. An example is the benchmark row table `experiments/unseen_cell_sop_20260914/closedloop_v251/src/bench_rows.csv`.

## Layout

| Folder | Contents |
|---|---|
| `analysis/` | Shared loaders and models: raw-data reader, ECM surfaces, EKF, LSTM voltage model, ridge SOH estimator |
| `experiments/unseen_cell_sop_20260914/` | Study scripts and every module they import |
| `paper_package_lsad_20261013_r3/` | Builds the fact table and the data tables used in the manuscript from the experiment outputs |
| `comparisons/20261014_baseline_pilot/` | Baselines that use the preceding pulse directly |
| `comparisons/20261104_ecm_rls/src/` | Equivalent circuit comparator with online recursive least-squares identification |
| `tables/` | `build_tables.py <repository root> <output dir>` builds Tables 1, 3, 4 and 5 |
| `figures/source/` | Builders of Figs. 1–5 (`build_fig1.py` to `build_fig5.py`) and their checks |

## Main scripts

These are in `experiments/unseen_cell_sop_20260914/`.

| Script | Role |
|---|---|
| `audit_rpcwby_aging_sop_v84.py`, `feasibility_audit_v106.py`, `diag_v261.py` | Definitions and audits of the reference SOP measurements |
| `closedloop_v251.py` | Requested-power scoring, where an ideal voltage clamp turns an over-request into an intervention |
| `closedloop_v253.py` | Per-row response model and simulation of the ideal voltage clamp |
| `closedloop_v255.py`, `closedloop_v256.py` | Scale calibration without the evaluated cell, over calibration targets of 4–6 % |
| `leakfree_v272.py` | Five local predictors and the leakage audit |
| `strict_v268.py` | Discharge calibration from models trained on the third cell alone |
| `cltrain_v274_features.py`, `specaudit_v287_features.py` | Features from the raw records |
| `specaudit_v287.py` | Local predictors after the specification audit: the manuscript configuration with fixed and adapted scales |
| `adaptive_v284.py` | Online adaptation of the derating scale |
| `compare_v278.py`, `s5_v286_posthoc.py` | Predictor implementation before the audit |
| `invariance_v276.py`, `posthoc_lsad_v320.py`, `posthoc_v321.py` | Information-flow checks and post hoc analyses |
| `selbias_v324.py` | Optimism from choosing among variants, and the resampling range |
| `chen_v265.py`, `chen_v266.py` | Reproduction of the LSTM-plus-search SOP method of Chen et al. |
| `presubmit_v281.py` | Pre-submission checks, including the common-scale check of the Chen reproduction |
| `mcu_port_v322/` | C port, STM32H563 firmware and host scripts for the microcontroller timing and memory measurements |

## Requirements

- Python 3.12. Install the packages with `pip install -r requirements.txt`.
- To build the firmware in `mcu_port_v322/fw` and `mcu_port_v322/fw_v322b`:
  - the GNU Arm toolchain (`arm-none-eabi-*`) on `PATH`, or the make variable `TOOLCHAIN` set to its `bin` directory;
  - the STM32CubeH5 firmware package unpacked as `mcu_port_v322/STM32Cube_FW_H5_V1.7.0`, or the make variable `FW_ROOT` set to its location.
- To run the board scripts, `STM32_Programmer_CLI` from STM32CubeProgrammer must be on `PATH`.
