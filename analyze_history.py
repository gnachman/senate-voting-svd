#!/usr/bin/env python3
"""
Historical Senate Voting Analysis with SVD

Generates a sequence of plots showing how Senate voting patterns
have evolved from the 80th Congress (1947) to present.

Data source: Voteview.com
https://voteview.com/data
"""

import sys
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARTIES = {
    100: ("Democrat", "blue"),
    200: ("Republican", "red"),
}

# Map congress number to years
def congress_to_years(congress: int) -> str:
    start_year = 1789 + (congress - 1) * 2
    return f"{start_year}-{start_year + 2}"


def load_all_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the bulk HSall files and filter to Senate only."""
    members = pd.read_csv(data_dir / "HSall_members.csv")
    votes = pd.read_csv(data_dir / "HSall_votes.csv")

    # Filter to Senate only
    members = members[members["chamber"] == "Senate"].copy()
    votes = votes[votes["chamber"] == "Senate"].copy()

    return members, votes


def build_vote_matrix(members: pd.DataFrame, votes: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Build the vote matrix for a single congress."""
    senator_ids = sorted(members["icpsr"].unique())
    roll_calls = sorted(votes["rollnumber"].unique())

    senator_idx = {sid: i for i, sid in enumerate(senator_ids)}
    roll_idx = {roll: i for i, roll in enumerate(roll_calls)}

    matrix = np.zeros((len(senator_ids), len(roll_calls)), dtype=np.float64)

    for _, row in votes.iterrows():
        sid = row["icpsr"]
        roll = row["rollnumber"]
        cast = row["cast_code"]

        if sid not in senator_idx:
            continue

        i = senator_idx[sid]
        j = roll_idx[roll]

        if cast in (1, 2, 3):
            matrix[i, j] = 1
        elif cast in (4, 5, 6):
            matrix[i, j] = -1

    senator_info = members.set_index("icpsr").loc[senator_ids].reset_index()
    return matrix, senator_info


def apply_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply SVD and return 2D coordinates and full singular values."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    coords = U[:, :2] * S[:2]
    return coords, S


def plot_congress(congress: int, coords: np.ndarray, senator_info: pd.DataFrame,
                  S: np.ndarray, ax: plt.Axes, show_names: bool = False):
    """Plot a single congress on the given axes."""
    for party_code, (party_name, color) in PARTIES.items():
        mask = senator_info["party_code"] == party_code
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=party_name,
            alpha=0.7,
            s=30,
        )

    # Variance explained
    total_var = np.sum(S**2)
    var_1 = S[0]**2 / total_var * 100
    var_2 = S[1]**2 / total_var * 100

    years = congress_to_years(congress)
    ax.set_title(f"{congress}th Congress ({years})\nDim1: {var_1:.0f}%, Dim2: {var_2:.0f}%",
                 fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def display_plot_iterm2(fig):
    """Display matplotlib figure using iTerm2 inline image protocol."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    image_data = base64.b64encode(buf.read()).decode("ascii")

    osc = "\033]"
    st = "\a"
    sys.stdout.write(f"{osc}1337;File=inline=1:{image_data}{st}\n")
    sys.stdout.flush()


def main():
    data_dir = Path(__file__).parent / "data"

    # Congress range: 80 (1947) to 118 (2023-2025)
    start_congress = 80
    end_congress = 118

    print("Loading data...")
    all_members, all_votes = load_all_data(data_dir)

    # Grid layout: 8 rows x 5 cols = 40 slots for 39 congresses
    n_cols = 5
    n_rows = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))
    axes = axes.flatten()

    congresses = list(range(start_congress, end_congress + 1))

    for i, congress in enumerate(congresses):
        print(f"  Processing {congress}th Congress ({congress_to_years(congress)})...")

        members = all_members[all_members["congress"] == congress]
        votes = all_votes[all_votes["congress"] == congress]

        if len(members) == 0 or len(votes) == 0:
            print(f"    No data for congress {congress}")
            continue

        matrix, senator_info = build_vote_matrix(members, votes)
        coords, S = apply_svd(matrix)

        # Ensure consistent orientation (Democrats on right)
        # Check if Democrats have negative mean x - if so, flip
        dem_mask = senator_info["party_code"] == 100
        if dem_mask.any() and coords[dem_mask, 0].mean() < 0:
            coords[:, 0] *= -1

        plot_congress(congress, coords, senator_info, S, axes[i])

    # Hide unused axes
    for j in range(len(congresses), len(axes)):
        axes[j].set_visible(False)

    # Add legend to first plot
    axes[0].legend(fontsize=6, loc="upper left")

    fig.suptitle("Evolution of Senate Polarization (1947-2025)", fontsize=14, y=0.995)
    plt.tight_layout()

    print("\nDisplaying plot...")
    display_plot_iterm2(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
