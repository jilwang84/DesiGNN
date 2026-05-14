# DesiGNN

Official implementation of **Proficient Graph Neural Network Design by Accumulating Knowledge on Large Language Models** (WSDM 2026).

DesiGNN designs GNN architectures for node classification by combining graph dataset understanding, dataset comparison, initial model suggestion, and knowledge-driven model proposal refinement. The release includes the paper method code, NAS-Bench-Graph transfer artifacts used by benchmarking runs, and dataset descriptions under `datasets/`.

## Environment

Python `3.10+` with `venv` and `pip` is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Set an OpenAI API key with one of the following:

```bash
export OPENAI_API_KEY=your_key
python main.py --openai_api_key your_key
```

or place the key in a local `key.txt` file.

## Usage

Run DesiGNN on Flickr:

```bash
python main.py --dataset Flickr --s 3 --k 30 --max_iter 30 --n_children 30 --seed 42
```

Run dataset comparison only:

```bash
python main.py --dataset Flickr --dcm_only --seed 42
```

Run formula-based dataset comparison:

```bash
python main.py --dataset Flickr --dcm_method formula --dcm_only --seed 42
```

Run a NAS-Bench-Graph benchmark target using the released benchmarking cache:

```bash
python main.py --dataset Planetoid:Cora --benchmarking --s 3 --max_iter 10 --seed 42
```

Run the same benchmark target and recompute LLM-based dataset similarity instead of reading the cache:

```bash
python main.py --dataset Planetoid:Cora --benchmarking --no_benchmark_cache --s 3 --max_iter 10 --seed 42
```

`--no_benchmark_cache` reruns the LLM-based dataset comparison for the current run and does not update `initial_model_benchmark.json`.

## Arguments

- `--dataset`: dataset identifier such as `Flickr`, `Actor`, `CitationFull:DBLP`, or `Planetoid:Cora`.
- `--benchmarking`: evaluate NAS-Bench-Graph benchmark targets through benchmark performance lookup.
- `--no_benchmark_cache`: skip the released benchmarking retrieval cache and run LLM-based dataset comparison.
- `--dcm_method`: choose `llm` or `formula` for dataset comparison.
- `--dcm_only`: run graph dataset understanding and dataset comparison without architecture refinement.
- `--seed`: seed for Python, NumPy, PyTorch, CUDA, graph sampling, and random search operations.
- `--max_iter`: number of model proposal refinement iterations.
- `--s`: number of retrieved benchmark datasets for initial model suggestion.
- `--k`: number of top NAS-Bench-Graph architectures retrieved per benchmark dataset.
- `--n_children`: number of controlled-exploration children generated per iteration.

## Released Artifacts

- `datasets/Benchmark datasets/`: benchmark dataset descriptions.
- `datasets/Unseen datasets/`: unseen dataset descriptions used by the paper experiments.
- `datasets/**/subgraphs/`: sampled subgraphs used for graph dataset understanding.
- `LLMConfiguredAutoGNN/initial_model_benchmark.json`: benchmarking retrieval cache for fast NAS-Bench-Graph runs.

## NAS-Bench-Graph

DesiGNN uses NAS-Bench-Graph for benchmark architecture-performance lookup. Please follow the official NAS-Bench-Graph repository for installation and data download:

[https://github.com/THUMNLab/NAS-Bench-Graph](https://github.com/THUMNLab/NAS-Bench-Graph)

The lightweight benchmark lookup is available through the `nas_bench_graph` Python package listed in `requirements.txt`:

```bash
python -m pip install nas_bench_graph
```

For complete per-epoch benchmark records, download the full `.bench` files from the NAS-Bench-Graph release link:

[https://figshare.com/articles/dataset/NAS-bench-Graph/20070371](https://figshare.com/articles/dataset/NAS-bench-Graph/20070371)

Place the downloaded `.bench` files in one directory and set:

```bash
export NBG_BENCH_DIR=/path/to/NBG
```

On Windows PowerShell:

```powershell
$env:NBG_BENCH_DIR="D:\path\to\NBG"
```

## Citation

```bibtex
@inproceedings{10.1145/3773966.3777982,
  author = {Wang, Jialiang and Liu, Hanmo and Di, Shimin and Wang, Zhili and Wang, Jiachuan and Chen, Lei and Zhou, Xiaofang},
  title = {Proficient Graph Neural Network Design by Accumulating Knowledge on Large Language Models},
  booktitle = {Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining},
  series = {WSDM '26},
  year = {2026},
  pages = {681--691},
  numpages = {11},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3773966.3777982},
  url = {https://doi.org/10.1145/3773966.3777982},
  keywords = {neural architecture search, graph neural networks, large language models}
}
```
