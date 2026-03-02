"""
analysis.py - High-level analysis queries on the Wikipedia graph

Computes interesting metrics and answers questions like:
  - What are the most "central" articles?
  - What's the shortest path between two articles?
  - What does the degree distribution look like?

Usage:
    python src/analysis.py --graph data/graph/ --pagerank results/pagerank/
"""

import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def create_spark():
    return (SparkSession.builder
            .appName("wikigraph-analysis")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def degree_distribution(vertices, outdir):
    """Plot in-degree and out-degree distributions (log-log scale)."""
    
    # collect degree data
    in_deg = (vertices.select("in_degree")
              .groupBy("in_degree")
              .count()
              .orderBy("in_degree")
              .toPandas())
    
    out_deg = (vertices.select("out_degree")
               .groupBy("out_degree")
               .count()
               .orderBy("out_degree")
               .toPandas())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # in-degree
    ax1.scatter(in_deg['in_degree'], in_deg['count'], 
                s=8, alpha=0.6, color='#2563eb')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('In-Degree', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('In-Degree Distribution (log-log)', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # out-degree
    ax2.scatter(out_deg['out_degree'], out_deg['count'],
                s=8, alpha=0.6, color='#dc2626')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Out-Degree', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Out-Degree Distribution (log-log)', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'degree_distribution.png'), dpi=150)
    plt.close()
    print("  Saved degree distribution plot")


def hub_authority_analysis(vertices, edges, outdir, top_k=20):
    """
    Simple hub/authority score approximation.
    
    Hubs = pages that link to many important pages (high out-degree to high in-degree targets)
    Authorities = pages that receive links from many hubs
    
    This is a simplified version of Kleinberg's HITS algorithm.
    """
    # authority score ~ weighted in-degree (weighted by source out-degree)
    source_out = (edges
                  .join(vertices.select("id", "out_degree"), 
                        edges["src"] == vertices["id"])
                  .select("src", "dst", "out_degree"))
    
    authority = (source_out
                 .groupBy("dst")
                 .agg(
                     F.count("*").alias("in_degree"),
                     F.sum("out_degree").alias("hub_weight")
                 )
                 .withColumn("authority_score", 
                             F.col("in_degree") * F.log1p(F.col("hub_weight")))
                 .orderBy(F.col("authority_score").desc()))
    
    print(f"\n  Top {top_k} Authority pages:")
    authority.select("dst", "in_degree", "authority_score").show(top_k, truncate=False)
    
    # hub score ~ out-degree to high-authority targets
    hub = (vertices
           .select("id", "out_degree", "in_degree")
           .withColumn("hub_score", 
                       F.col("out_degree") * F.log1p(F.col("in_degree")))
           .orderBy(F.col("hub_score").desc()))
    
    print(f"\n  Top {top_k} Hub pages:")
    hub.select("id", "out_degree", "hub_score").show(top_k, truncate=False)


def bfs_shortest_path(edges, source, target, max_depth=6):
    """
    BFS shortest path between two articles.
    
    Classic "six degrees of Wikipedia"; most articles are reachable
    in 3-5 clicks.
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    
    print(f"\n  Finding path: {source} -> {target}")
    
    # frontier starts at source
    frontier = spark.createDataFrame([(source, [source])], ["node", "path"])
    visited = {source}
    
    for depth in range(max_depth):
        # expand frontier
        expanded = (frontier
                    .join(edges, frontier["node"] == edges["src"])
                    .select(
                        edges["dst"].alias("node"),
                        F.concat(F.col("path"), F.array(F.col("dst"))).alias("path")
                    ))
        
        # check if target reached
        found = expanded.filter(F.col("node") == target)
        if found.count() > 0:
            path = found.first()["path"]
            print(f"  Found path (length {len(path) - 1}): {' -> '.join(path)}")
            return path
        
        # filter visited
        new_nodes = [row["node"] for row in expanded.select("node").distinct().collect()
                     if row["node"] not in visited]
        
        if not new_nodes:
            print(f"  No path found within {max_depth} hops")
            return None
        
        visited.update(new_nodes)
        frontier = expanded.filter(F.col("node").isin(new_nodes))
        print(f"    Depth {depth + 1}: frontier size = {len(new_nodes)}")
    
    print(f"  No path found within {max_depth} hops")
    return None


def main():
    parser = argparse.ArgumentParser(description='Graph analysis')
    parser.add_argument('--graph', required=True, help='Graph parquet path')
    parser.add_argument('--pagerank', help='PageRank results path (optional)')
    parser.add_argument('--output', default='results/', help='Output dir')
    parser.add_argument('--path-from', help='Source for shortest path query')
    parser.add_argument('--path-to', help='Target for shortest path query')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    spark = create_spark()
    
    try:
        vertices = spark.read.parquet(f"{args.graph}/vertices")
        edges = spark.read.parquet(f"{args.graph}/edges")
        
        print("=== Degree Distribution ===")
        degree_distribution(vertices, args.output)
        
        print("\n=== Hub/Authority Analysis ===")
        hub_authority_analysis(vertices, edges, args.output)
        
        if args.path_from and args.path_to:
            print("\n=== Shortest Path ===")
            bfs_shortest_path(edges, args.path_from, args.path_to)
        
        if args.pagerank:
            print("\n=== PageRank + Degree Correlation ===")
            pr = spark.read.parquet(args.pagerank)
            combined = (vertices
                        .join(pr, "id")
                        .select("id", "in_degree", "out_degree", "rank"))
            
            print("  Top pages by each metric:")
            for col_name in ["rank", "in_degree", "out_degree"]:
                print(f"\n  --- Top 10 by {col_name} ---")
                combined.orderBy(F.col(col_name).desc()).show(10, truncate=False)
    
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
