# wikigraph

Large-scale analysis of the Wikipedia link graph using PySpark. Computes PageRank, finds connected components, and characterizes the degree distribution of the world's largest encyclopedia.

The question that started this: "How many clicks does it really take to get from any random Wikipedia article to Philosophy?" (Spoiler: it's usually 3-5, and the degree distribution follows a power law, which is pretty wild for something built by millions of anonymous editors.)

## Quick Start

```bash
# generate synthetic test graph (no download needed)
python scripts/generate_synthetic.py data/edges.csv 5000

# build graph (creates parquet files)
spark-submit src/build_graph.py --edges data/edges.csv --output data/graph/

# run PageRank
spark-submit src/pagerank.py --graph data/graph/ --iterations 20 --output results/pagerank/

# connected components
spark-submit src/components.py --graph data/graph/ --output results/components/

# analysis + plots
spark-submit src/analysis.py --graph data/graph/ --pagerank results/pagerank/ --output results/
```

### Using real Wikipedia data

```bash
# downloads Simple English Wikipedia (~100MB), good for testing
bash scripts/download_dump.sh simplewiki

# parse the SQL dump into edge list
python scripts/parse_dump.py data/simplewiki-latest-pagelinks.sql.gz data/edges.csv

# for full English Wikipedia (~6GB compressed), use:
# bash scripts/download_dump.sh enwiki
```

## What it does

### PageRank
Iterative PageRank computation over the full link graph. Implemented in pure PySpark (no GraphFrames dependency) so it runs without extra package installs. The output tells you which Wikipedia articles are the most "important" by link structure.

### Connected Components
Weakly connected components via label propagation. Reveals the structure of Wikipedia's link graph: how many isolated clusters exist, and how large the giant component is.

### Analysis
- **Degree distribution**: Log-log plot showing the power-law nature of Wikipedia's link structure
- **Hub/Authority scores**: Simplified HITS algorithm for finding hub pages (link to many important articles) vs authority pages (linked to by many hubs)
- **Shortest path**: BFS path finder between any two articles ("six degrees of Wikipedia")

## Architecture

```
scripts/
├── download_dump.sh       # Fetch Wikipedia SQL dumps
├── parse_dump.py          # Parse SQL → CSV edge list
└── generate_synthetic.py  # Power-law synthetic graph generator
src/
├── build_graph.py         # Edge CSV → Spark DataFrames (parquet)
├── pagerank.py            # Iterative PageRank
├── components.py          # Connected components (label propagation)
└── analysis.py            # Degree distribution, HITS, BFS, correlation
```

## Requirements

- Python 3.8+
- PySpark 3.x (`pip install pyspark`)
- matplotlib, numpy (for plots)

Tested on Spark 3.5 with local mode. For the full English Wikipedia dump you'll want at least 16GB RAM and a few cores.

## Graph Format

Internally the graph is stored as two parquet directories:
- `vertices/`: `id` (page title), `in_degree`, `out_degree`
- `edges/`: `src`, `dst` (both page titles)

You can swap in any CSV with `source,target` columns.

## Interesting Findings (Synthetic Graph)

On a 5,000-node synthetic graph with preferential attachment:
- The degree distribution closely follows a power law (α ≈ 2.1)
- The giant connected component contains ~98% of all pages
- PageRank converges in about 15 iterations
- Top PageRank pages correlate strongly with high in-degree, but not perfectly. Some low-degree pages rank high because they're linked to by other high-ranking pages
