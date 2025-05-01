<pre><code>project_3/
├── data/                       # MUTAG dataset
│
├── models/
│   ├── erdos_renyi.py          # Code to generate baseline graphs
│   ├── gnn_generator.py        # Deep generative model code
│   └── __init__.py
│
├── evaluation/
│   ├── weisfeiler_lehman.py    # Code for WL test for graph uniqueness
│   ├── statistics.py           # Degree, clustering coefficient, eigenvector centrality
│   └── __init__.py
│
├── scripts/
│   └── evaluate_graphs.py
│
├── utils/
│   └── dataset_loader.py       # Load MUTAG dataset
</code></pre>
