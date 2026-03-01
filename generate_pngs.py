#!/usr/bin/env python3
"""
Generate individual PNG plots for each Congress.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARTIES = {
    100: ("Democrat", "blue"),
    200: ("Republican", "red"),
}


def congress_to_years(congress: int) -> str:
    start_year = 1789 + (congress - 1) * 2
    return f"{start_year}-{start_year + 2}"


def load_all_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    members = pd.read_csv(data_dir / "HSall_members.csv")
    votes = pd.read_csv(data_dir / "HSall_votes.csv")
    members = members[members["chamber"] == "Senate"].copy()
    votes = votes[votes["chamber"] == "Senate"].copy()
    return members, votes


def build_vote_matrix(members: pd.DataFrame, votes: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
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
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    coords = U[:, :2] * S[:2]
    return coords, S


def plot_congress(congress: int, coords: np.ndarray, senator_info: pd.DataFrame,
                  S: np.ndarray, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 12))

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
            s=50,
        )

        # Add senator names
        for idx in np.where(mask)[0]:
            name = senator_info.iloc[idx]["bioname"].split(",")[0]
            ax.annotate(
                name,
                (coords[idx, 0], coords[idx, 1]),
                fontsize=5,
                alpha=0.7,
            )

    total_var = np.sum(S**2)
    var_1 = S[0]**2 / total_var * 100
    var_2 = S[1]**2 / total_var * 100

    years = congress_to_years(congress)
    ax.set_title(f"{congress}th Congress ({years}) - SVD of Senate Voting Records\n"
                 f"Dim1: {var_1:.1f}%, Dim2: {var_2:.1f}%", fontsize=14)
    ax.set_xlabel("First Dimension")
    ax.set_ylabel("Second Dimension")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    data_dir = Path(__file__).parent / "data"
    plots_dir = Path(__file__).parent / "plots"

    start_congress = 80
    end_congress = 118

    print("Loading data...")
    all_members, all_votes = load_all_data(data_dir)

    for congress in range(start_congress, end_congress + 1):
        print(f"  {congress}th Congress...", end=" ", flush=True)

        members = all_members[all_members["congress"] == congress]
        votes = all_votes[all_votes["congress"] == congress]

        if len(members) == 0 or len(votes) == 0:
            print("skipped (no data)")
            continue

        matrix, senator_info = build_vote_matrix(members, votes)
        coords, S = apply_svd(matrix)

        # Consistent orientation (Democrats on right)
        dem_mask = senator_info["party_code"] == 100
        if dem_mask.any() and coords[dem_mask, 0].mean() < 0:
            coords[:, 0] *= -1

        output_path = plots_dir / f"congress_{congress:03d}.png"
        plot_congress(congress, coords, senator_info, S, output_path)
        print(f"saved")

    print(f"\nDone! {end_congress - start_congress + 1} PNGs saved to {plots_dir}/")


if __name__ == "__main__":
    main()
