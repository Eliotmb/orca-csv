import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from ml.data import ORCACSVGeometryDataset
from ml.model import ORCAGNN


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


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_mae += mae(pred.detach(), batch.y.detach().view(-1)) * batch.num_graphs
        total_rmse += rmse(pred.detach(), batch.y.detach().view(-1)) * batch.num_graphs
        n += batch.num_graphs

    return {
        "loss": total_loss / n,
        "mae": total_mae / n,
        "rmse": total_rmse / n,
    }


def eval_epoch(model, loader, device):
    model.eval()
    total_mae = 0.0
    total_rmse = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            total_mae += mae(pred, batch.y.view(-1)) * batch.num_graphs
            total_rmse += rmse(pred, batch.y.view(-1)) * batch.num_graphs
            n += batch.num_graphs
    return {
        "mae": total_mae / n,
        "rmse": total_rmse / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a GNN on ORCA CSV data")
    parser.add_argument("--coords_csv", required=True, help="Path to ORCA_Atomic_Coordinates.csv")
    parser.add_argument("--energies_csv", required=True, help="Path to ORCA_Energies.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", default="./ml/cache", help="Cache directory for processed dataset")
    parser.add_argument("--log_dir", default="./ml/logs", help="Directory to save metrics and plots")
    parser.add_argument("--plots", action="store_true", help="Save MAE/RMSE plots (requires matplotlib)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.root).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset = ORCACSVGeometryDataset(
        root=args.root,
        coords_csv=args.coords_csv,
        energies_csv=args.energies_csv,
    )

    train_idx, val_idx, test_idx = split_dataset(dataset)
    train_ds = dataset[list(train_idx)]
    val_ds = dataset[list(val_idx)]
    test_ds = dataset[list(test_idx)]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ORCAGNN(node_in_dim=4, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(
        args.device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mae = float("inf")
    best_state = None

    history = {"epoch": [], "train_loss": [], "train_mae": [], "train_rmse": [], "val_mae": [], "val_rmse": []}

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, args.device)
        val_metrics = eval_epoch(model, val_loader, args.device)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["train_rmse"].append(train_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"train MAE {train_metrics['mae']:.4f} "
            f"train RMSE {train_metrics['rmse']:.4f} | "
            f"val MAE {val_metrics['mae']:.4f} "
            f"val RMSE {val_metrics['rmse']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = eval_epoch(model, test_loader, args.device)
    print(f"Test MAE: {test_metrics['mae']:.4f} | Test RMSE: {test_metrics['rmse']:.4f}")

    out_path = Path("./ml") / "gnn_energy_regressor.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model weights to {out_path}")

    metrics_path = log_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_mae,train_rmse,val_mae,val_rmse\n")
        for i in range(len(history["epoch"])):
            f.write(
                f"{history['epoch'][i]},{history['train_loss'][i]:.6f},{history['train_mae'][i]:.6f},"
                f"{history['train_rmse'][i]:.6f},{history['val_mae'][i]:.6f},{history['val_rmse'][i]:.6f}\n"
            )
    print(f"Saved metrics to {metrics_path}")

    if args.plots:
        if plt is None:
            print("matplotlib not installed; skipping plots. Install matplotlib or omit --plots.")
        else:
            _make_plots(history, log_dir)


def _make_plots(history: Dict[str, list], log_dir: Path):
    # MAE plot
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_mae"], label="train MAE")
    plt.plot(history["epoch"], history["val_mae"], label="val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (Hartree)")
    plt.legend()
    plt.tight_layout()
    mae_path = log_dir / "mae_curve.png"
    plt.savefig(mae_path, dpi=150)
    plt.close()

    # RMSE plot
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_rmse"], label="train RMSE")
    plt.plot(history["epoch"], history["val_rmse"], label="val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (Hartree)")
    plt.legend()
    plt.tight_layout()
    rmse_path = log_dir / "rmse_curve.png"
    plt.savefig(rmse_path, dpi=150)
    plt.close()

    print(f"Saved plots to {mae_path} and {rmse_path}")


if __name__ == "__main__":
    main()
