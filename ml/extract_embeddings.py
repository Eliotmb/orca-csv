import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

from ml.data import ORCACSVGeometryDataset
from ml.model import ORCAGNN


def extract_embeddings(model, loader, device):
    """Extract graph-level embeddings before the final prediction head."""
    model.eval()
    embeddings = []
    labels = []
    section_ids = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Forward through GNN layers to get node embeddings
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
            x = model.proj(x)
            for conv in model.convs:
                x = conv(x, edge_index)
                x = torch.nn.functional.relu(x)
            # Global pooling to get graph-level embeddings
            from torch_geometric.nn import global_mean_pool
            graph_emb = global_mean_pool(x, batch_idx)
            
            embeddings.append(graph_emb.cpu().numpy())
            labels.append(batch.y.cpu().numpy())
            section_ids.extend([int(d.section_id) for d in batch.to_data_list()])
    
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels, section_ids


def compute_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity between embeddings."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    # Cosine similarity matrix
    similarity = normalized @ normalized.T
    return similarity


def visualize_embeddings_2d(embeddings, labels, section_ids, method='tsne', output_dir=None):
    """Reduce embeddings to 2D and visualize."""
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        coords_2d = reducer.fit_transform(embeddings)
        title = 't-SNE projection of molecular embeddings'
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
        var = reducer.explained_variance_ratio_
        title = f'PCA projection (var: {var[0]:.2%}, {var[1]:.2%})'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap='viridis', 
                        alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Annotate a few points with section IDs
    step = max(1, len(section_ids) // 20)
    for i in range(0, len(section_ids), step):
        ax.annotate(f'{section_ids[i]}', (coords_2d[i, 0], coords_2d[i, 1]),
                   fontsize=7, alpha=0.7)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy (Hartree)', rotation=270, labelpad=20)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(title)
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f'embedding_{method}.png'
        plt.savefig(output_path, dpi=150)
        print(f'Saved {method.upper()} plot to {output_path}')
    else:
        plt.show()
    plt.close()


def visualize_similarity_matrix(similarity, section_ids, output_dir=None):
    """Plot heatmap of pairwise similarity."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xlabel('Geometry Section')
    ax.set_ylabel('Geometry Section')
    ax.set_title('Cosine Similarity between Molecular Graphs')
    
    # Show a subset of ticks if too many samples
    n = len(section_ids)
    tick_step = max(1, n // 20)
    tick_indices = list(range(0, n, tick_step))
    tick_labels = [section_ids[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'similarity_matrix.png'
        plt.savefig(output_path, dpi=150)
        print(f'Saved similarity matrix to {output_path}')
    else:
        plt.show()
    plt.close()


def find_most_similar(similarity, section_ids, top_k=5):
    """Find most similar pairs of molecules."""
    n = similarity.shape[0]
    # Mask diagonal and lower triangle to avoid duplicates
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    masked_sim = np.where(mask, similarity, -np.inf)
    
    # Get top-k most similar pairs
    flat_indices = np.argsort(masked_sim.ravel())[::-1][:top_k]
    pairs = np.unravel_index(flat_indices, (n, n))
    
    print(f"\nTop {top_k} most similar geometry pairs (cosine similarity):")
    for i, j, idx in zip(pairs[0], pairs[1], range(top_k)):
        sim_val = similarity[i, j]
        print(f"  {idx+1}. Section {section_ids[i]} ↔ Section {section_ids[j]}: {sim_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Extract and visualize GNN embeddings")
    parser.add_argument("--coords_csv", required=True, help="Path to ORCA_Atomic_Coordinates.csv")
    parser.add_argument("--energies_csv", required=True, help="Path to ORCA_Energies.csv")
    parser.add_argument("--model_path", default="./ml/gnn_energy_regressor.pt", help="Path to trained model weights")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--root", default="./ml/cache", help="Cache directory")
    parser.add_argument("--output_dir", default="./ml/embeddings", help="Output directory for plots")
    parser.add_argument("--top_k", type=int, default=10, help="Number of most similar pairs to show")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = ORCACSVGeometryDataset(
        root=args.root,
        coords_csv=args.coords_csv,
        energies_csv=args.energies_csv,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load trained model
    model = ORCAGNN(node_in_dim=4, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=0.0).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device, weights_only=True))
    print(f"Loaded model from {args.model_path}")

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, labels, section_ids = extract_embeddings(model, loader, args.device)
    print(f"Extracted {len(embeddings)} graph embeddings with dimension {embeddings.shape[1]}")

    # Save embeddings
    emb_path = output_dir / "embeddings.npz"
    np.savez(emb_path, embeddings=embeddings, labels=labels, section_ids=section_ids)
    print(f"Saved embeddings to {emb_path}")

    # Compute similarity
    print("Computing similarity matrix...")
    similarity = compute_similarity_matrix(embeddings)

    # Find most similar pairs
    find_most_similar(similarity, section_ids, top_k=args.top_k)

    # Visualize
    print("Generating visualizations...")
    visualize_embeddings_2d(embeddings, labels, section_ids, method='tsne', output_dir=output_dir)
    visualize_embeddings_2d(embeddings, labels, section_ids, method='pca', output_dir=output_dir)
    visualize_similarity_matrix(similarity, section_ids, output_dir=output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
