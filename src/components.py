"""
components.py - Connected components analysis on the Wikipedia graph

Finds weakly connected components using iterative label propagation.
This tells you which articles are reachable from each other, and
identifies isolated clusters.

Usage:
    spark-submit src/components.py --graph data/graph/ --output results/components/
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def create_spark():
    return (SparkSession.builder
            .appName("wikigraph-components")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def connected_components(spark, graph_path, max_iter=30):
    """
    Weakly connected components via label propagation.
    
    Each vertex starts with its own label. In each iteration, every vertex
    adopts the minimum label among itself and all its neighbors. This
    converges when labels stop changing.
    
    We treat the graph as undirected for this (weak connectivity).
    """
    edges = spark.read.parquet(f"{graph_path}/edges")
    vertices = spark.read.parquet(f"{graph_path}/vertices")
    
    # assign numeric IDs for label propagation
    # (string min is fine but slower than numeric)
    vertex_ids = (vertices.select("id")
                  .withColumn("vid", F.monotonically_increasing_id())
                  .cache())
    
    # build undirected edge list with numeric IDs
    edges_with_ids = (edges
                      .join(vertex_ids, edges["src"] == vertex_ids["id"])
                      .select(F.col("vid").alias("src_vid"), "dst")
                      .join(vertex_ids, F.col("dst") == vertex_ids["id"])
                      .select("src_vid", F.col("vid").alias("dst_vid")))
    
    # make undirected: add reverse edges
    forward = edges_with_ids.select(
        F.col("src_vid").alias("v1"),
        F.col("dst_vid").alias("v2"))
    reverse = edges_with_ids.select(
        F.col("dst_vid").alias("v1"),
        F.col("src_vid").alias("v2"))
    undirected = forward.union(reverse).dropDuplicates()
    
    # initialize labels: each vertex is its own component
    labels = vertex_ids.select(
        F.col("vid"),
        F.col("vid").alias("component"))
    
    n_vertices = vertex_ids.count()
    print(f"Running connected components on {n_vertices} vertices")
    
    for iteration in range(max_iter):
        # propagate: each vertex takes min(own label, neighbors' labels)
        neighbor_labels = (undirected
                           .join(labels, undirected["v2"] == labels["vid"])
                           .select(
                               undirected["v1"].alias("vid"),
                               F.col("component").alias("neighbor_comp")
                           ))
        
        new_labels = (labels
                      .select("vid", F.col("component").alias("self_comp"))
                      .join(
                          neighbor_labels.groupBy("vid")
                          .agg(F.min("neighbor_comp").alias("min_neighbor")),
                          "vid", "left")
                      .fillna({"min_neighbor": 2**62})
                      .select(
                          "vid",
                          F.least("self_comp", "min_neighbor").alias("component")
                      ))
        
        # check convergence
        changed = (labels.alias("old")
                   .join(new_labels.alias("new"), "vid")
                   .filter(F.col("old.component") != F.col("new.component"))
                   .count())
        
        labels = new_labels
        
        print(f"  Iteration {iteration + 1}: {changed} labels changed")
        
        if changed == 0:
            print(f"  Converged after {iteration + 1} iterations")
            break
    
    # join back to original titles
    result = (labels
              .join(vertex_ids, "vid")
              .select("id", "component"))
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Find connected components')
    parser.add_argument('--graph', required=True, help='Graph parquet path')
    parser.add_argument('--output', default='results/components/')
    parser.add_argument('--max-iter', type=int, default=30)
    args = parser.parse_args()
    
    spark = create_spark()
    try:
        components = connected_components(spark, args.graph, args.max_iter)
        
        # component size distribution
        comp_sizes = (components
                      .groupBy("component")
                      .agg(F.count("*").alias("size"))
                      .orderBy(F.col("size").desc()))
        
        n_components = comp_sizes.count()
        print(f"\n  Found {n_components} connected components")
        
        print("\n  Largest components:")
        comp_sizes.show(20, truncate=False)
        
        # save
        components.write.mode("overwrite").parquet(args.output)
        print(f"  Results saved to {args.output}")
        
        # summary CSV
        (comp_sizes
         .coalesce(1)
         .write.mode("overwrite")
         .option("header", "true")
         .csv(f"{args.output}_summary"))
        
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
