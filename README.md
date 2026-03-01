# Senate Voting Analysis with SVD

Visualize U.S. Senate voting patterns using Singular Value Decomposition (SVD). This project replicates the classic analysis that plots senators in 2D ideological space based on their roll call votes.

## Data Source

All voting data comes from [Voteview](https://voteview.com/data), the Poole-Rosenthal project that has tracked congressional voting since the 1st Congress.

## Setup

```bash
pip install -r requirements.txt
```

## Download Data

For single-congress analysis (118th Congress, 2023-2025):

```bash
mkdir -p data
curl -o data/S118_members.csv "https://voteview.com/static/data/out/members/S118_members.csv"
curl -o data/S118_votes.csv "https://voteview.com/static/data/out/votes/S118_votes.csv"
```

For historical analysis (all congresses):

```bash
curl -o data/HSall_members.csv "https://voteview.com/static/data/out/members/HSall_members.csv"
curl -o data/HSall_votes.csv "https://voteview.com/static/data/out/votes/HSall_votes.csv"
```

## Scripts

### analyze.py

Analyze a single congress (118th by default). Displays the plot inline using iTerm2's image protocol.

```bash
python analyze.py
```

### analyze_history.py

Generate a grid visualization showing all congresses from the 80th (1947) to present.

```bash
python analyze_history.py
```

### generate_pngs.py

Generate individual PNG files for each congress, saved to `plots/`.

```bash
python generate_pngs.py
```

### index.html

Mobile-friendly web viewer for the generated plots. Supports swipe gestures and keyboard navigation.

```bash
python -m http.server 8000
# Open http://localhost:8000
```

## How It Works

1. Build a vote matrix: rows are senators, columns are roll calls, values are +1 (yea), -1 (nay), or 0 (abstain/absent)
2. Apply SVD: `A = U S Vᵀ`
3. Plot senators using coordinates from `U @ S` (first 2 columns)

The first dimension typically captures left-right ideology. The second dimension has historically captured regional (North-South) differences, though this has diminished as parties have polarized.

## Key Insight

`U @ S = A @ V` — the senator coordinates are their voting records projected onto the principal vote patterns (columns of V).
