import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from ml.data_multitask import ORCACSVMultiTaskDataset
from ml.model_multitask import ORCAGNNMultiTask


def mae(pred, target):
    return (pred - target).abs().mean().item()


def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)).item()


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=(1 - train_ratio), random_state=seed, shuffle=True)
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - val_size), random_state=seed, shuffle=True)
    return train_idx, val_idx, test_idx


def train_one_epoch(model, loader, optimizer, device, energy_weight=1.0, dipole_weight=1.0):
    model.train()
    total_loss = 0.0
    total_energy_mae = 0.0
    total_dipole_mae = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        energy_pred, dipole_pred = model(batch)
        
        energy_loss = F.mse_loss(energy_pred, batch.y_energy.view(-1))
        dipole_loss = F.mse_loss(dipole_pred, batch.y_dipole.view(-1))
        loss = energy_weight * energy_loss + dipole_weight * dipole_loss
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_energy_mae += mae(energy_pred.detach(), batch.y_energy.detach().view(-1)) * batch.num_graphs
        total_dipole_mae += mae(dipole_pred.detach(), batch.y_dipole.detach().view(-1)) * batch.num_graphs
        n += batch.num_graphs

    return {
        "loss": total_loss / n,
        "energy_mae": total_energy_mae / n,
        "dipole_mae": total_dipole_mae / n,
    }


def eval_epoch(model, loader, device):
    model.eval()
    total_energy_mae = 0.0
    total_energy_rmse = 0.0
    total_dipole_mae = 0.0
    total_dipole_rmse = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            energy_pred, dipole_pred = model(batch)
            
            total_energy_mae += mae(energy_pred, batch.y_energy.view(-1)) * batch.num_graphs
            total_energy_rmse += rmse(energy_pred, batch.y_energy.view(-1)) * batch.num_graphs
            total_dipole_mae += mae(dipole_pred, batch.y_dipole.view(-1)) * batch.num_graphs
            total_dipole_rmse += rmse(dipole_pred, batch.y_dipole.view(-1)) * batch.num_graphs
            n += batch.num_graphs
    return {
        "energy_mae": total_energy_mae / n,
        "energy_rmse": total_energy_rmse / n,
        "dipole_mae": total_dipole_mae / n,
        "dipole_rmse": total_dipole_rmse / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Train multi-task GNN on ORCA CSV data")
    parser.add_argument("--coords_csv", required=True, help="Path to ORCA_Atomic_Coordinates.csv")
    parser.add_argument("--energies_csv", required=True, help="Path to ORCA_Energies.csv")
    parser.add_argument("--properties_csv", required=True, help="Path to ORCA_Molecular_Properties.csv with dipole")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--energy_weight", type=float, default=1.0, help="Weight for energy loss")
    parser.add_argument("--dipole_weight", type=float, default=0.1, help="Weight for dipole loss")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", default="./ml/cache_multitask", help="Cache directory")
    parser.add_argument("--log_dir", default="./ml/logs_multitask", help="Directory to save metrics and plots")
    parser.add_argument("--plots", action="store_true", help="Save plots")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.root).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset = ORCACSVMultiTaskDataset(
        root=args.root,
        coords_csv=args.coords_csv,
        energies_csv=args.energies_csv,
        properties_csv=args.properties_csv,
    )

    train_idx, val_idx, test_idx = split_dataset(dataset)
    train_ds = dataset[list(train_idx)]
    val_ds = dataset[list(val_idx)]
    test_ds = dataset[list(test_idx)]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ORCAGNNMultiTask(
        node_in_dim=4, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None

    history = {
        "epoch": [],
        "train_loss": [],
        "train_energy_mae": [],
        "train_dipole_mae": [],
        "val_energy_mae": [],
        "val_dipole_mae": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, args.device, args.energy_weight, args.dipole_weight
        )
        val_metrics = eval_epoch(model, val_loader, args.device)

        val_loss = val_metrics["energy_mae"] + val_metrics["dipole_mae"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_energy_mae"].append(train_metrics["energy_mae"])
        history["train_dipole_mae"].append(train_metrics["dipole_mae"])
        history["val_energy_mae"].append(val_metrics["energy_mae"])
        history["val_dipole_mae"].append(val_metrics["dipole_mae"])

        print(
            f"Epoch {epoch:03d} | "
            f"loss {train_metrics['loss']:.4f} "
            f"E_MAE {train_metrics['energy_mae']:.4f} "
            f"D_MAE {train_metrics['dipole_mae']:.4f} | "
            f"val E_MAE {val_metrics['energy_mae']:.4f} "
            f"val D_MAE {val_metrics['dipole_mae']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = eval_epoch(model, test_loader, args.device)
    print(
        f"Test | Energy MAE: {test_metrics['energy_mae']:.4f} RMSE: {test_metrics['energy_rmse']:.4f} | "
        f"Dipole MAE: {test_metrics['dipole_mae']:.4f} RMSE: {test_metrics['dipole_rmse']:.4f}"
    )

    out_path = Path("./ml") / "gnn_multitask_regressor.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model weights to {out_path}")

    metrics_path = log_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_energy_mae,train_dipole_mae,val_energy_mae,val_dipole_mae\n")
        for i in range(len(history["epoch"])):
            f.write(
                f"{history['epoch'][i]},{history['train_loss'][i]:.6f},"
                f"{history['train_energy_mae'][i]:.6f},{history['train_dipole_mae'][i]:.6f},"
                f"{history['val_energy_mae'][i]:.6f},{history['val_dipole_mae'][i]:.6f}\n"
            )
    print(f"Saved metrics to {metrics_path}")

    if args.plots and plt is not None:
        _make_plots(history, log_dir)


def _make_plots(history: Dict[str, list], log_dir: Path):
    # Energy MAE plot
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_energy_mae"], label="train")
    plt.plot(history["epoch"], history["val_energy_mae"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Energy MAE (Hartree)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_dir / "energy_mae_curve.png", dpi=150)
    plt.close()

    # Dipole MAE plot
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_dipole_mae"], label="train")
    plt.plot(history["epoch"], history["val_dipole_mae"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Dipole MAE (Debye)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_dir / "dipole_mae_curve.png", dpi=150)
    plt.close()

    print(f"Saved plots to {log_dir}")


if __name__ == "__main__":
    main()
