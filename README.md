project_3/
│
├── data/                # MUTAG dataset
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
│
├── scripts/					# Code for running each part of the project
│   ├── run_erdos_renyi.py
│   ├── run_gnn_generator.py
│   ├── evaluate_graphs.py
│
├── utils/
│   └── dataset_loader.py       # Load MUTAG dataset and convert to networkx if needed
│
├── requirements.txt
├── README.md
└── config.yaml                 # Parameters like number of graphs, model configs, etc.