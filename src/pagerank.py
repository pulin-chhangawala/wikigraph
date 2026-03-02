"""
pagerank.py - Run PageRank on the Wikipedia link graph

Implements PageRank using iterative Spark DataFrame operations.
We don't depend on GraphFrames here so the project runs without
extra package installs, just vanilla PySpark.

Usage:
    spark-submit src/pagerank.py --graph data/graph/ --iterations 20 --output results/pagerank/
    python src/pagerank.py --graph data/graph/ --iterations 20 --output results/pagerank/
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def create_spark():
    return (SparkSession.builder
            .appName("wikigraph-pagerank")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def run_pagerank(spark, graph_path, num_iterations=20, damping=0.85):
    """
    Iterative PageRank implementation.
    
    PR(p) = (1-d)/N + d * sum(PR(q)/L(q) for q linking to p)
    
    where d is the damping factor and L(q) is the out-degree of q.
    
    This is essentially what Google used to rank web pages in 1998.
    The intuition: a page is important if other important pages link to it.
    """
    edges = spark.read.parquet(f"{graph_path}/edges")
    vertices = spark.read.parquet(f"{graph_path}/vertices")
    
    n_pages = vertices.count()
    print(f"Running PageRank on {n_pages} pages, {num_iterations} iterations")
    print(f"  Damping factor: {damping}")
    
    # initialize: every page starts with rank = 1/N
    ranks = vertices.select("id").withColumn("rank", F.lit(1.0 / n_pages))
    
    # precompute out-degrees
    out_degrees = (edges.groupBy("src")
                   .agg(F.count("*").alias("out_degree"))
                   .cache())
    
    for iteration in range(num_iterations):
        # compute contributions: each page distributes its rank
        # evenly among all pages it links to
        contribs = (edges
                    .join(ranks, edges["src"] == ranks["id"])
                    .join(out_degrees, edges["src"] == out_degrees["src"])
                    .select(
                        edges["dst"].alias("id"),
                        (F.col("rank") / F.col("out_degree")).alias("contrib")
                    ))
        
        # aggregate contributions for each page
        new_ranks = (contribs
                     .groupBy("id")
                     .agg(F.sum("contrib").alias("total_contrib")))
        
        # apply damping factor
        # pages with no incoming links get the random surfer probability
        ranks = (vertices.select("id")
                 .join(new_ranks, "id", "left")
                 .fillna(0, subset=["total_contrib"])
                 .select(
                     "id",
                     ((1 - damping) / n_pages + damping * F.col("total_contrib"))
                     .alias("rank")
                 ))
        
        if (iteration + 1) % 5 == 0 or iteration == 0:
            # checkpoint progress
            max_rank = ranks.agg(F.max("rank")).collect()[0][0]
            print(f"  Iteration {iteration + 1}/{num_iterations}, "
                  f"max rank: {max_rank:.8f}")
    
    return ranks


def main():
    parser = argparse.ArgumentParser(description='Run PageRank')
    parser.add_argument('--graph', required=True, help='Graph parquet path')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--damping', type=float, default=0.85)
    parser.add_argument('--output', default='results/pagerank/')
    parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args()
    
    spark = create_spark()
    try:
        ranks = run_pagerank(spark, args.graph, args.iterations, args.damping)
        
        # show top results
        print(f"\n  Top {args.top_k} pages by PageRank:")
        top_pages = (ranks
                     .orderBy(F.col("rank").desc())
                     .limit(args.top_k))
        top_pages.show(args.top_k, truncate=False)
        
        # save full results
        ranks.write.mode("overwrite").parquet(args.output)
        print(f"  Full rankings saved to {args.output}")
        
        # also save top-k as readable CSV
        (top_pages
         .coalesce(1)
         .write.mode("overwrite")
         .option("header", "true")
         .csv(f"{args.output}_top{args.top_k}"))
        print(f"  Top {args.top_k} saved as CSV")
        
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
