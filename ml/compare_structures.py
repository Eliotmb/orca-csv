import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from ml.data import ORCACSVGeometryDataset


def parse_orca_file(out_file: Path):
    """Parse ORCA .out file for coordinates, energies, and dipole."""
    text = out_file.read_text(encoding='utf-16', errors='ignore')
    
    # Coordinates
    coord_pattern = re.compile(
        r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*-+\s*((?:\s*[A-Z][a-z]?\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\n)+)",
        re.MULTILINE
    )
    coords_rows = []
    section = 0
    for m in coord_pattern.finditer(text):
        section += 1
        atom_idx = 0
        for line in m.group(1).strip().splitlines():
            parts = line.split()
            if len(parts) == 4:
                atom_idx += 1
                element, x, y, z = parts
                coords_rows.append([section, atom_idx, element, float(x), float(y), float(z)])
    
    # Energies
    energy_pattern = re.compile(r"Total Energy\s*:\s*(-?\d+\.\d+)")
    energies_rows = [[i+1, float(m.group(1))] for i, m in enumerate(energy_pattern.finditer(text))]
    
    # Dipole magnitude (final value)
    dipole_pattern = re.compile(r"Magnitude \(Debye\)\s*:\s*(-?\d+\.\d+)")
    dipole_matches = list(dipole_pattern.finditer(text))
    dipole_value = float(dipole_matches[-1].group(1)) if dipole_matches else 0.0
    
    return coords_rows, energies_rows, dipole_value, section


def compare_structures(file1: Path, file2: Path, model_path: Path, device: str, hidden_dim: int = 128, num_layers: int = 3):
    """Compare two ORCA output files using trained GNN embeddings."""
    from ml.model import ORCAGNN
    
    # Parse both files
    print(f"Parsing {file1.name}...")
    coords1, energies1, dipole1, sections1 = parse_orca_file(file1)
    print(f"  Found {len(coords1)} coord rows, {len(energies1)} energies, dipole={dipole1:.2f}")
    
    print(f"Parsing {file2.name}...")
    coords2, energies2, dipole2, sections2 = parse_orca_file(file2)
    print(f"  Found {len(coords2)} coord rows, {len(energies2)} energies, dipole={dipole2:.2f}")
    
    # Create temporary CSVs
    tmp_dir = Path('./ml/tmp_compare')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (coords, energies) in enumerate([(coords1, energies1), (coords2, energies2)], 1):
        coords_df = pd.DataFrame(coords, columns=['Geometry_Section', 'Atom_Number', 'Element', 'X_Angstrom', 'Y_Angstrom', 'Z_Angstrom'])
        energies_df = pd.DataFrame(energies, columns=['Iteration', 'Total_Energy_Hartree'])
        coords_df.to_csv(tmp_dir / f'coords_{idx}.csv', index=False)
        energies_df.to_csv(tmp_dir / f'energies_{idx}.csv', index=False)
    
    # Load model
    model = ORCAGNN(node_in_dim=4, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Extract embeddings
    embeddings = []
    for idx in [1, 2]:
        dataset = ORCACSVGeometryDataset(
            root=str(tmp_dir / f'cache_{idx}'),
            coords_csv=str(tmp_dir / f'coords_{idx}.csv'),
            energies_csv=str(tmp_dir / f'energies_{idx}.csv'),
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        file_embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
                x = model.proj(x)
                for conv in model.convs:
                    x = conv(x, edge_index)
                    x = torch.nn.functional.relu(x)
                from torch_geometric.nn import global_mean_pool
                graph_emb = global_mean_pool(x, batch_idx)
                file_embeddings.append(graph_emb.cpu().numpy())
        
        import numpy as np
        file_embeddings = np.vstack(file_embeddings)
        # Average all geometries per file
        avg_emb = file_embeddings.mean(axis=0, keepdims=True)
        embeddings.append(avg_emb)
    
    # Compute similarity
    import numpy as np
    emb1, emb2 = embeddings[0], embeddings[1]
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    cosine_sim = (emb1 @ emb2.T) / (norm1 * norm2 + 1e-8)
    euclidean_dist = np.linalg.norm(emb1 - emb2)
    
    print(f"\n{'='*60}")
    print(f"Structure Comparison Results")
    print(f"{'='*60}")
    print(f"File 1: {file1.name}")
    print(f"  Geometries: {sections1}, Final Energy: {energies1[-1][1]:.6f} Ha, Dipole: {dipole1:.2f} D")
    print(f"File 2: {file2.name}")
    print(f"  Geometries: {sections2}, Final Energy: {energies2[-1][1]:.6f} Ha, Dipole: {dipole2:.2f} D")
    print(f"\nEmbedding-based Similarity:")
    print(f"  Cosine Similarity: {cosine_sim[0,0]:.4f} (1.0 = identical, -1.0 = opposite)")
    print(f"  Euclidean Distance: {euclidean_dist:.4f} (lower = more similar)")
    print(f"\nProperty Differences:")
    print(f"  ΔEnergy: {abs(energies1[-1][1] - energies2[-1][1]):.6f} Ha")
    print(f"  ΔDipole: {abs(dipole1 - dipole2):.4f} D")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Compare two ORCA output files")
    parser.add_argument("--file1", required=True, help="First ORCA .out file")
    parser.add_argument("--file2", required=True, help="Second ORCA .out file")
    parser.add_argument("--model_path", default="./ml/gnn_energy_regressor.pt", help="Trained model weights")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    file1 = Path(args.file1)
    file2 = Path(args.file2)
    model_path = Path(args.model_path)
    
    if not file1.exists():
        raise FileNotFoundError(f"File not found: {file1}")
    if not file2.exists():
        raise FileNotFoundError(f"File not found: {file2}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    compare_structures(file1, file2, model_path, args.device, args.hidden_dim, args.num_layers)


if __name__ == "__main__":
    main()
