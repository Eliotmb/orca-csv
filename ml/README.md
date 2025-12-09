# ORCA GNN Training

This folder contains a minimal PyTorch Geometric pipeline to regress total energies from ORCA CSV exports.

## Data expected
- `ORCA_Atomic_Coordinates.csv` with columns: `Geometry_Section, Atom_Number, Element, X_Angstrom, Y_Angstrom, Z_Angstrom`.
- `ORCA_Energies.csv` with columns: `Iteration, Total_Energy_Hartree`.
- Each `Geometry_Section` becomes one graph; the target energy is matched by `Iteration` for that section.

## Install (CPU example)
```bash
pip install -r requirements-ml.txt
```
For GPU, install matching PyTorch/PyG wheels first (per https://pytorch.org/), then run the same requirements file.

## Train
```bash
python -m ml.train --coords_csv ORCA_Atomic_Coordinates.csv --energies_csv ORCA_Energies.csv \
  --epochs 30 --batch_size 8 --lr 1e-3 --plots
```
Flags:
- `--plots` saves MAE/RMSE curves (requires matplotlib).
- `--log_dir` (default `./ml/logs`) stores `metrics.csv`, `mae_curve.png`, `rmse_curve.png`.
- Model weights saved to `ml/gnn_energy_regressor.pt`.

## Notes
- Node features: `[Z, x, y, z]`; edges are fully connected (no self-loops).
- Coordinates are centered per graph to improve stability.
- Metrics reported: MAE and RMSE on val/test splits.
