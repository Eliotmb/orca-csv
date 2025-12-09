import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset


# Very small periodic table helper; extend as needed.
_ATOMIC_NUMBERS: Dict[str, int] = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


def element_to_z(symbol: str) -> int:
    z = _ATOMIC_NUMBERS.get(symbol)
    if z is None:
        raise ValueError(f"Element '{symbol}' not in lookup table; extend _ATOMIC_NUMBERS")
    return z


def build_edge_index(num_nodes: int, fully_connected: bool = True) -> torch.Tensor:
    # Fully connected undirected graph without self-loops.
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)
    rows: List[int] = []
    cols: List[int] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            rows.extend([i, j])
            cols.extend([j, i])
    return torch.tensor([rows, cols], dtype=torch.long)


def load_energy_targets(energies_csv: Path) -> Dict[int, float]:
    df = pd.read_csv(energies_csv)
    # Expect columns: Iteration, Total_Energy_Hartree
    if not {"Iteration", "Total_Energy_Hartree"}.issubset(df.columns):
        raise ValueError("energies CSV must include columns: Iteration, Total_Energy_Hartree")
    return {int(r.Iteration): float(r.Total_Energy_Hartree) for r in df.itertuples()}


def load_coordinates(coords_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(coords_csv)
    expected = {"Geometry_Section", "Atom_Number", "Element", "X_Angstrom", "Y_Angstrom", "Z_Angstrom"}
    if not expected.issubset(df.columns):
        raise ValueError(f"coords CSV must include columns: {sorted(expected)}")
    return df


class ORCACSVGeometryDataset(InMemoryDataset):
    """Graph dataset built from ORCA CSV exports.

    Each geometry section becomes one graph. Node features: atomic number + xyz (Å).
    Target: total energy for the matching iteration (from energies CSV).
    """

    def __init__(
        self,
        root: str,
        coords_csv: str,
        energies_csv: str,
        transform=None,
        pre_transform=None,
    ):
        self.coords_csv = Path(coords_csv)
        self.energies_csv = Path(energies_csv)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.coords_csv.name, self.energies_csv.name]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        # Copy user-provided CSVs into raw_dir so InMemoryDataset's checks pass.
        self.raw_dir_path = Path(self.raw_dir)
        self.raw_dir_path.mkdir(parents=True, exist_ok=True)
        (self.raw_dir_path / self.coords_csv.name).write_bytes(self.coords_csv.read_bytes())
        (self.raw_dir_path / self.energies_csv.name).write_bytes(self.energies_csv.read_bytes())

    def process(self):
        coords_df = load_coordinates(self.coords_csv)
        energy_map = load_energy_targets(self.energies_csv)

        data_list: List[Data] = []
        for section_id, section_df in coords_df.groupby("Geometry_Section"):
            num_nodes = len(section_df)
            if num_nodes == 0:
                continue
            if section_id not in energy_map:
                # Skip if we do not have a matching energy target.
                continue

            z = torch.tensor([element_to_z(sym) for sym in section_df["Element"]], dtype=torch.float)
            pos = torch.tensor(
                section_df[["X_Angstrom", "Y_Angstrom", "Z_Angstrom"]].to_numpy(), dtype=torch.float
            )
            # Optional: center coordinates to improve stability.
            pos = pos - pos.mean(dim=0, keepdim=True)

            edge_index = build_edge_index(num_nodes)
            x = torch.cat([z.unsqueeze(-1), pos], dim=1)  # shape [N, 4]
            y = torch.tensor([energy_map[section_id]], dtype=torch.float)

            data = Data(x=x, pos=pos, edge_index=edge_index, y=y)
            data.section_id = int(section_id)
            data.num_nodes = num_nodes
            data_list.append(data)

        if len(data_list) == 0:
            raise ValueError("No graphs built; ensure Geometry_Section matches Iteration in energies CSV")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
