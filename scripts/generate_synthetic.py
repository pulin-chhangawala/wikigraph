"""
generate_synthetic.py - Generate a synthetic Wikipedia-like link graph

Creates a power-law-distributed directed graph that mimics the link structure
of Wikipedia. Useful for testing when you don't want to download the full dump.

Usage:
    python3 generate_synthetic.py data/edges.csv [num_pages]
"""

import csv
import random
import sys
import math

# roughly mirrors the distribution of Wikipedia categories
CATEGORIES = [
    "Science", "History", "Geography", "Technology", "Mathematics",
    "Biology", "Physics", "Chemistry", "Philosophy", "Politics",
    "Art", "Music", "Literature", "Sports", "Economics",
    "Medicine", "Engineering", "Computer_Science", "Psychology", "Sociology"
]

PREFIXES = [
    "Introduction_to", "History_of", "List_of", "Outline_of",
    "", "", "", "", "",  # bias toward no prefix
]

SUFFIXES = [
    "_theory", "_system", "_method", "_analysis", "_problem",
    "", "", "", "", "", "", "",  # bias toward no suffix
]

SPECIFIC_TERMS = {
    "Science": ["Hypothesis", "Experiment", "Observation", "Peer_review", "Scientific_method"],
    "History": ["Ancient_Rome", "World_War_II", "Renaissance", "Industrial_Revolution", "Cold_War"],
    "Geography": ["Pacific_Ocean", "Mount_Everest", "Amazon_River", "Sahara", "Antarctica"],
    "Technology": ["Internet", "Artificial_intelligence", "Blockchain", "Quantum_computing", "Robotics"],
    "Mathematics": ["Calculus", "Linear_algebra", "Number_theory", "Graph_theory", "Topology"],
    "Biology": ["DNA", "Evolution", "Cell_biology", "Genetics", "Ecology"],
    "Physics": ["Quantum_mechanics", "Relativity", "Thermodynamics", "Electromagnetism", "Optics"],
    "Computer_Science": ["Algorithm", "Data_structure", "Machine_learning", "Operating_system", "Compiler"],
}


def generate_page_titles(n_pages):
    """Generate fake but plausible Wikipedia article titles."""
    titles = set()
    
    # add category pages
    for cat in CATEGORIES:
        titles.add(cat)
    
    # add specific terms
    for terms in SPECIFIC_TERMS.values():
        titles.update(terms)
    
    # generate more
    while len(titles) < n_pages:
        cat = random.choice(CATEGORIES)
        prefix = random.choice(PREFIXES)
        suffix = random.choice(SUFFIXES)
        base = f"{cat}_{random.randint(1, 1000)}"
        title = f"{prefix}{base}{suffix}".strip("_")
        titles.add(title)
    
    return list(titles)[:n_pages]


def generate_edges(titles, avg_links_per_page=15):
    """
    Generate directed edges with a power-law out-degree distribution.
    
    Important pages (low index = "older" = more established) get more
    incoming links, which is roughly how Wikipedia works.
    """
    n = len(titles)
    edges = []
    
    for i in range(n):
        # power-law out-degree: most pages link to ~10-20 others,
        # some hub pages link to hundreds
        out_degree = max(1, int(random.paretovariate(1.5) * avg_links_per_page / 3))
        out_degree = min(out_degree, n // 4)  # cap at 25% of total
        
        # preferential attachment: bias toward linking to "important" pages
        # (lower indices = more incoming links)
        targets = set()
        while len(targets) < out_degree:
            # mix of preferential (70%) and random (30%)
            if random.random() < 0.7:
                # preferential: power-law biased toward low indices
                target = int(random.betavariate(1, 3) * n)
            else:
                target = random.randint(0, n - 1)
            
            if target != i:
                targets.add(target)
        
        for t in targets:
            edges.append((titles[i], titles[t]))
    
    return edges


def main():
    outpath = sys.argv[1] if len(sys.argv) > 1 else "data/edges.csv"
    n_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    random.seed(42)
    
    print(f"Generating synthetic graph: {n_pages} pages...")
    titles = generate_page_titles(n_pages)
    edges = generate_edges(titles)
    
    print(f"  {len(titles)} pages, {len(edges)} edges")
    print(f"  Avg out-degree: {len(edges)/len(titles):.1f}")
    
    with open(outpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'target'])
        w.writerows(edges)
    
    print(f"  Written to {outpath}")


if __name__ == '__main__':
    main()
