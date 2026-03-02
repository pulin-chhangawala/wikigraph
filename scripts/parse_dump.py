#!/usr/bin/env python3
"""
parse_dump.py - Parse a Wikipedia SQL dump into a clean edge list CSV

Reads the pagelinks SQL dump (gzipped) and outputs a CSV with
(source_title, target_title) columns.

Usage:
    python3 parse_dump.py data/simplewiki-latest-pagelinks.sql.gz data/edges.csv
"""

import gzip
import re
import csv
import sys


def parse_sql_inserts(filepath):
    """
    Parse INSERT statements from a MySQL dump file.
    
    The pagelinks table has columns:
      (pl_from, pl_namespace, pl_title, pl_from_namespace)
    
    We only care about pl_from (source page ID) and pl_title (target page title)
    in the main namespace (namespace 0).
    """
    open_fn = gzip.open if filepath.endswith('.gz') else open
    
    insert_re = re.compile(r"INSERT INTO `pagelinks` VALUES\s*")
    # match individual tuples: (123,0,'Page_title',0)
    tuple_re = re.compile(r"\((\d+),(\d+),'((?:[^'\\]|\\.)*)',(\d+)\)")
    
    edges = []
    n_lines = 0
    
    with open_fn(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            n_lines += 1
            if n_lines % 50000 == 0:
                print(f"  Processing line {n_lines}, {len(edges)} edges so far...",
                      end='\r')
            
            if not insert_re.match(line):
                continue
            
            for match in tuple_re.finditer(line):
                pl_from = int(match.group(1))
                pl_namespace = int(match.group(2))
                pl_title = match.group(3)
                pl_from_ns = int(match.group(4))
                
                # only main namespace (0)
                if pl_namespace == 0 and pl_from_ns == 0:
                    # unescape MySQL strings
                    pl_title = pl_title.replace("\\'", "'")
                    pl_title = pl_title.replace("\\\\", "\\")
                    edges.append((pl_from, pl_title))
    
    print(f"\n  Parsed {len(edges)} edges from {n_lines} lines")
    return edges


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 parse_dump.py <input.sql.gz> <output.csv>")
        sys.exit(1)
    
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    
    print(f"Parsing {inpath}...")
    edges = parse_sql_inserts(inpath)
    
    # note: source is a page ID (int), target is a title (string)
    # for the graph analysis we'll handle this mapping in Spark
    with open(outpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source_id', 'target_title'])
        w.writerows(edges)
    
    print(f"Written {len(edges)} edges to {outpath}")


if __name__ == '__main__':
    main()
