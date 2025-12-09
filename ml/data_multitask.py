import math
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset


_ATOMIC_NUMBERS: Dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}


def element_to_z(symbol: str) -> int:
    z = _ATOMIC_NUMBERS.get(symbol)
    if z is None:
        raise ValueError(f"Element '{symbol}' not in lookup table")
    return z


def build_edge_index(num_nodes: int) -> torch.Tensor:
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            rows.extend([i, j])
            cols.extend([j, i])
    return torch.tensor([rows, cols], dtype=torch.long)


class ORCACSVMultiTaskDataset(InMemoryDataset):
    """Graph dataset with energy and dipole moment targets."""

    def __init__(
        self,
        root: str,
        coords_csv: str,
        energies_csv: str,
        properties_csv: str,
        transform=None,
        pre_transform=None,
    ):
        self.coords_csv = Path(coords_csv)
        self.energies_csv = Path(energies_csv)
        self.properties_csv = Path(properties_csv)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.coords_csv.name, self.energies_csv.name, self.properties_csv.name]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        self.raw_dir_path = Path(self.raw_dir)
        self.raw_dir_path.mkdir(parents=True, exist_ok=True)
        (self.raw_dir_path / self.coords_csv.name).write_bytes(self.coords_csv.read_bytes())
        (self.raw_dir_path / self.energies_csv.name).write_bytes(self.energies_csv.read_bytes())
        (self.raw_dir_path / self.properties_csv.name).write_bytes(self.properties_csv.read_bytes())

    def process(self):
        coords_df = pd.read_csv(self.coords_csv)
        energies_df = pd.read_csv(self.energies_csv)
        properties_df = pd.read_csv(self.properties_csv)

        # Build energy map
        energy_map = {int(r.Iteration): float(r.Total_Energy_Hartree) for r in energies_df.itertuples()}
        
        # Build dipole map (properties CSV has one row per final geometry)
        dipole_map = {}
        if 'Geometry_Section' in properties_df.columns and 'Dipole_Magnitude' in properties_df.columns:
            for r in properties_df.itertuples():
                dipole_map[int(r.Geometry_Section)] = float(r.Dipole_Magnitude)
        elif 'Dipole_Magnitude' in properties_df.columns:
            # If only one row, assign to last section
            dipole_map[max(coords_df['Geometry_Section'])] = float(properties_df['Dipole_Magnitude'].iloc[0])

        data_list: List[Data] = []
        for section_id, section_df in coords_df.groupby("Geometry_Section"):
            num_nodes = len(section_df)
            if num_nodes == 0 or section_id not in energy_map:
                continue

            z = torch.tensor([element_to_z(sym) for sym in section_df["Element"]], dtype=torch.float)
            pos = torch.tensor(
                section_df[["X_Angstrom", "Y_Angstrom", "Z_Angstrom"]].to_numpy(), dtype=torch.float
            )
            pos = pos - pos.mean(dim=0, keepdim=True)

            edge_index = build_edge_index(num_nodes)
            x = torch.cat([z.unsqueeze(-1), pos], dim=1)
            
            y_energy = torch.tensor([energy_map[section_id]], dtype=torch.float)
            y_dipole = torch.tensor([dipole_map.get(section_id, 0.0)], dtype=torch.float)

            data = Data(x=x, pos=pos, edge_index=edge_index, y_energy=y_energy, y_dipole=y_dipole)
            data.section_id = int(section_id)
            data.num_nodes = num_nodes
            data_list.append(data)

        if len(data_list) == 0:
            raise ValueError("No graphs built")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
