"""
build_graph.py - Load edge list into Spark and build a GraphFrame

Reads the CSV edge list (from parse_dump or generate_synthetic) and
creates a directed graph using GraphFrames. Saves the graph as parquet
for downstream analysis.

Usage:
    spark-submit --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
        src/build_graph.py --edges data/edges.csv --output data/graph/

    # or for local testing:
    python src/build_graph.py --edges data/edges.csv --output data/graph/
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType


def create_spark(app_name="wikigraph"):
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def build_graph(spark, edges_path, output_path):
    """
    Build a vertex + edge DataFrame pair from the CSV edge list.
    
    The CSV should have 'source' and 'target' columns (both are page titles).
    """
    print(f"Loading edges from {edges_path}...")
    
    edges_df = (spark.read
                .option("header", "true")
                .csv(edges_path)
                .select(
                    F.col("source").alias("src"),
                    F.col("target").alias("dst")
                ))
    
    # deduplicate edges
    edges_df = edges_df.dropDuplicates()
    
    n_edges = edges_df.count()
    print(f"  {n_edges} unique edges loaded")
    
    # extract unique vertices from both sides
    src_vertices = edges_df.select(F.col("src").alias("id"))
    dst_vertices = edges_df.select(F.col("dst").alias("id"))
    vertices_df = src_vertices.union(dst_vertices).dropDuplicates()
    
    n_vertices = vertices_df.count()
    print(f"  {n_vertices} unique vertices")
    
    # compute basic degree stats while we're here
    out_degrees = (edges_df.groupBy("src")
                   .agg(F.count("*").alias("out_degree")))
    in_degrees = (edges_df.groupBy("dst")
                  .agg(F.count("*").alias("in_degree")))
    
    # join degrees onto vertices
    vertices_df = (vertices_df
                   .join(out_degrees, vertices_df["id"] == out_degrees["src"], "left")
                   .drop("src")
                   .join(in_degrees, vertices_df["id"] == in_degrees["dst"], "left")
                   .drop("dst")
                   .fillna(0))
    
    # save as parquet
    vertices_df.write.mode("overwrite").parquet(f"{output_path}/vertices")
    edges_df.write.mode("overwrite").parquet(f"{output_path}/edges")
    
    print(f"  Graph saved to {output_path}/")
    
    # print some basic stats
    print("\n  --- Graph Statistics ---")
    print(f"  Vertices: {n_vertices}")
    print(f"  Edges:    {n_edges}")
    print(f"  Avg out-degree: {n_edges / max(n_vertices, 1):.1f}")
    
    # top pages by in-degree
    print("\n  Top 10 pages by in-degree:")
    (vertices_df
     .orderBy(F.col("in_degree").desc())
     .select("id", "in_degree", "out_degree")
     .show(10, truncate=False))
    
    return vertices_df, edges_df


def main():
    parser = argparse.ArgumentParser(description='Build graph from edge list')
    parser.add_argument('--edges', required=True, help='CSV edge list path')
    parser.add_argument('--output', default='data/graph/', help='Output path')
    args = parser.parse_args()
    
    spark = create_spark()
    try:
        build_graph(spark, args.edges, args.output)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
