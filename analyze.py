#!/usr/bin/env python3
"""
Senate Voting Analysis with SVD

Data source: Voteview.com (118th Congress, 2023-2025)
https://voteview.com/data

Vote codes:
  1 = Yea, 2 = Paired Yea, 3 = Announced Yea
  4 = Announced Nay, 5 = Paired Nay, 6 = Nay
  7, 8 = Present
  9 = Not Voting
  0 = Not a member during this vote
"""

import sys
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Party codes from Voteview
PARTIES = {
    100: ("Democrat", "blue"),
    200: ("Republican", "red"),
    328: ("Independent", "purple"),  # Sinema
    # Add others as needed
}


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the members and votes CSVs."""
    members = pd.read_csv(data_dir / "S118_members.csv")
    votes = pd.read_csv(data_dir / "S118_votes.csv")
    return members, votes


def build_vote_matrix(members: pd.DataFrame, votes: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build the vote matrix for SVD.

    Returns:
        matrix: Shape (n_senators, n_votes) with values:
                 1 = Yea, -1 = Nay, 0 = abstain/absent/not member
        senator_info: DataFrame with senator metadata, indexed same as matrix rows
    """
    # Get unique senators and roll calls
    senator_ids = sorted(members["icpsr"].unique())
    roll_calls = sorted(votes["rollnumber"].unique())

    # Create mapping from IDs to indices
    senator_idx = {sid: i for i, sid in enumerate(senator_ids)}
    roll_idx = {roll: i for i, roll in enumerate(roll_calls)}

    # Initialize matrix with zeros (abstain/absent)
    matrix = np.zeros((len(senator_ids), len(roll_calls)), dtype=np.float64)

    # Fill in votes
    for _, row in votes.iterrows():
        sid = row["icpsr"]
        roll = row["rollnumber"]
        cast = row["cast_code"]

        if sid not in senator_idx:
            continue

        i = senator_idx[sid]
        j = roll_idx[roll]

        # Convert cast_code to vote value
        if cast in (1, 2, 3):  # Yea variants
            matrix[i, j] = 1
        elif cast in (4, 5, 6):  # Nay variants
            matrix[i, j] = -1
        # else: leave as 0 (abstain/absent/not member)

    # Build senator info DataFrame aligned with matrix rows
    senator_info = members.set_index("icpsr").loc[senator_ids].reset_index()

    return matrix, senator_info


def apply_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply SVD to the vote matrix.

    The vote matrix has shape (n_senators, n_votes).
    You want to find a low-dimensional representation of senators.

    np.linalg.svd() returns U, S, Vt where:
      - U has shape (n_senators, n_senators) - left singular vectors
      - S has shape (min(n_senators, n_votes),) - singular values
      - Vt has shape (n_votes, n_votes) - right singular vectors

    For plotting senators in 2D, you'll want the first 2 columns of U,
    scaled by the corresponding singular values.

    Returns:
        U: Left singular vectors
        S: Singular values
        Vt: Right singular vectors (transposed)
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    U_2 = U[:, :2]
    S_2 = S[:2]
    Vt_2 = Vt[:2, :]
    return (U_2, S_2, Vt_2)


def get_2d_coordinates(U: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Extract 2D coordinates for plotting senators.

    The first two columns of U @ diag(S) give you the 2D projection.
    Or equivalently: U[:, :2] * S[:2]

    Returns:
        coords: Shape (n_senators, 2) array of x, y coordinates
    """
    return U*S


def plot_senators(coords: np.ndarray, senator_info: pd.DataFrame, show_names: bool = False):
    """
    Plot senators in 2D space, colored by party.
    Uses iTerm2 inline image protocol.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

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

        if show_names:
            for idx in np.where(mask)[0]:
                name = senator_info.iloc[idx]["bioname"].split(",")[0]
                ax.annotate(
                    name,
                    (coords[idx, 0], coords[idx, 1]),
                    fontsize=6,
                    alpha=0.7,
                )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("First Dimension (typically left-right ideology)")
    ax.set_ylabel("Second Dimension")
    ax.set_title("118th Senate (2023-2025) - SVD of Voting Records")
    ax.legend()

    # Render to iTerm2 inline image
    display_plot_iterm2(fig)
    plt.close(fig)


def display_plot_iterm2(fig):
    """Display matplotlib figure using iTerm2 inline image protocol."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    image_data = base64.b64encode(buf.read()).decode("ascii")

    # iTerm2 proprietary escape sequence for inline images
    # https://iterm2.com/documentation-images.html
    osc = "\033]"
    st = "\a"

    sys.stdout.write(f"{osc}1337;File=inline=1:{image_data}{st}\n")
    sys.stdout.flush()


def main():
    data_dir = Path(__file__).parent / "data"

    print("Loading data...")
    members, votes = load_data(data_dir)
    print(f"  {len(members)} senators, {votes['rollnumber'].nunique()} roll calls")

    print("Building vote matrix...")
    matrix, senator_info = build_vote_matrix(members, votes)
    print(f"  Matrix shape: {matrix.shape}")

    # Show vote participation stats
    participation = (matrix != 0).mean(axis=1)
    print(f"  Average participation: {participation.mean():.1%}")

    print("\nApplying SVD...")
    U, S, Vt = apply_svd(matrix)

    print("Extracting 2D coordinates...")
    coords = get_2d_coordinates(U, S)

    print("Plotting...")
    plot_senators(coords, senator_info, show_names=True)

    # Print some stats about the SVD
    total_variance = np.sum(S**2)
    explained_1 = S[0]**2 / total_variance
    explained_2 = S[1]**2 / total_variance
    print(f"\nVariance explained:")
    print(f"  Dim 1: {explained_1:.1%}")
    print(f"  Dim 2: {explained_2:.1%}")
    print(f"  Total (2D): {explained_1 + explained_2:.1%}")


if __name__ == "__main__":
    main()
