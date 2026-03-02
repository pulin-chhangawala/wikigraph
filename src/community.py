"""
community.py - Community detection in the Wikipedia link graph

Implements Label Propagation for community detection. Each node
starts with its own label, then iteratively adopts the most common
label among its neighbors. Converges to stable communities.

Also includes modularity calculation to score the quality of
detected communities.

Usage:
    spark-submit src/community.py --input data/graph --output results/communities
"""

import argparse
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def label_propagation(spark, edges_path, max_iter=20):
    """
    Label Propagation Algorithm for community detection.
    
    Each vertex starts with a unique label (its ID). On each
    iteration, every vertex adopts the most frequent label
    among its neighbors. Iterates until convergence.
    """
    edges = spark.read.parquet(edges_path)
    
    # initialize: each vertex gets its own label
    vertices = (edges.select(F.col('src').alias('id'))
                .union(edges.select(F.col('dst').alias('id')))
                .distinct()
                .withColumn('label', F.col('id')))
    
    print(f"  Starting label propagation on {vertices.count()} vertices, "
          f"{edges.count()} edges")
    
    for i in range(max_iter):
        # join edges with current labels
        labeled_edges = (edges
                        .join(vertices.withColumnRenamed('label', 'nbr_label'),
                              edges.dst == vertices.id, 'inner')
                        .select(edges.src, F.col('nbr_label')))
        
        # for each vertex, find the most common neighbor label
        new_labels = (labeled_edges
                     .groupBy('src', 'nbr_label')
                     .count()
                     .withColumn('rank', F.row_number().over(
                         Window.partitionBy('src')
                         .orderBy(F.desc('count'), 'nbr_label')))
                     .filter(F.col('rank') == 1)
                     .select(F.col('src').alias('id'),
                            F.col('nbr_label').alias('new_label')))
        
        # update labels
        updated = (vertices
                  .join(new_labels, 'id', 'left')
                  .withColumn('label',
                             F.coalesce(F.col('new_label'), F.col('label')))
                  .select('id', 'label'))
        
        # check convergence
        changed = (updated.join(vertices, 'id')
                  .filter(updated.label != vertices.label)
                  .count())
        
        vertices = updated.cache()
        
        n_communities = vertices.select('label').distinct().count()
        print(f"  Iteration {i+1}: {changed} labels changed, "
              f"{n_communities} communities")
        
        if changed == 0:
            print(f"  Converged after {i+1} iterations")
            break
    
    return vertices


def compute_modularity(spark, edges_path, communities):
    """
    Compute modularity Q of the community assignment.
    
    Q = (1/2m) * Σ [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)
    
    Higher Q (closer to 1) = better community structure.
    Typically Q > 0.3 indicates significant structure.
    """
    edges = spark.read.parquet(edges_path)
    m = edges.count()  # total edges
    
    # degree of each vertex
    degrees = (edges.groupBy('src').count()
              .withColumnRenamed('count', 'degree')
              .withColumnRenamed('src', 'id'))
    
    # join communities with edges (both endpoints must match)
    comm_edges = (edges
                 .join(communities.withColumnRenamed('label', 'c_src'),
                       edges.src == communities.id)
                 .join(communities.withColumnRenamed('label', 'c_dst')
                       .withColumnRenamed('id', 'id2'),
                       edges.dst == F.col('id2')))
    
    # count intra-community edges
    intra = comm_edges.filter(F.col('c_src') == F.col('c_dst')).count()
    
    # expected edges (sum of k_i * k_j for same community pairs)
    comm_degree = (degrees
                  .join(communities, 'id')
                  .groupBy('label')
                  .agg(F.sum('degree').alias('total_degree')))
    
    expected = (comm_degree
               .withColumn('expected', 
                          F.col('total_degree') * F.col('total_degree') / (2.0 * m))
               .agg(F.sum('expected')).collect()[0][0])
    
    Q = (intra / (2.0 * m)) - (expected / (4.0 * m * m))
    
    return Q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/graph/edges')
    parser.add_argument('--output', default='results/communities')
    parser.add_argument('--max-iter', type=int, default=20)
    args = parser.parse_args()
    
    spark = (SparkSession.builder
            .appName('wikigraph-community')
            .getOrCreate())
    
    communities = label_propagation(spark, args.input, args.max_iter)
    
    # stats
    stats = (communities.groupBy('label')
            .count()
            .orderBy(F.desc('count')))
    
    print("\n  Top 10 communities:")
    stats.show(10)
    
    total_communities = stats.count()
    print(f"  Total communities: {total_communities}")
    
    # save
    communities.write.mode('overwrite').parquet(args.output)
    print(f"  Saved to {args.output}")
    
    spark.stop()


if __name__ == '__main__':
    main()
