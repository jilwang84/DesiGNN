# Copyright (c) 2026 DesiGNN Authors
# License: Apache-2.0 license

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from GraphDatasetComparison.GraphDatasetComparison import GraphDatasetComparison
from GraphDatasetUnderstanding.DatasetReader import DatasetReader
from GraphDatasetUnderstanding.GraphDatasetUnderstanding import GraphDatasetUnderstanding
from LLMConfiguredAutoGNN.LLMConfiguredAutoGNN import LLMConfiguredAutoGNN


REPO_ROOT = Path(__file__).resolve().parent

BENCHMARK_MAPPING = {
    "Cora": "Planetoid:Cora",
    "CiteSeer": "Planetoid:CiteSeer",
    "PubMed": "Planetoid:PubMed",
    "CS": "Coauthor:CS",
    "Physics": "Coauthor:Physics",
    "Photo": "Amazon:Photo",
    "Computers": "Amazon:Computers",
    "ogbn-arxiv": "ogbn-arxiv",
    "ogbn-proteins": "ogbn-proteins",
}

DEFAULT_BENCHMARKS = list(BENCHMARK_MAPPING.values())

ALL_METRICS = [
    "average_clustering_coefficient",
    "local_average_betweenness_centrality",
    "density",
    "average_degree_centrality",
    "local_average_closeness_centrality",
    "average_degree",
    "edge_count",
    "local_graph_diameter",
    "local_average_shortest_path_length",
    "assortativity",
    "average_eigenvector_centrality",
    "feature_dimensionality",
    "node_count",
    "node_feature_diversity",
    "connected_components",
    "label_homophily",
]


def canonical_name(dataset_identifier: str) -> str:
    return dataset_identifier.split(":", 1)[1] if ":" in dataset_identifier else dataset_identifier


def as_source_root(path: Path) -> str:
    text = str(path)
    return text if text.endswith(os.sep) else text + os.sep


def read_text(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_default_user_description(unseen_dir: str, dataset_name: str) -> str:
    path = os.path.join(unseen_dir, canonical_name(dataset_name), "user_description.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return (
        "The user does not provide a description for the dataset, please understand the dataset "
        "based entirely on the following graph topological features."
    )


def resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key.strip()
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    key_file = REPO_ROOT / "key.txt"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return None


def get_llm(api_key: Optional[str], model: str):
    key = resolve_api_key(api_key)
    if not key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(api_key=key, temperature=0, model=model)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_benchmark_cache(cache_path: Path, dataset_name: str, top_s: int) -> Optional[Tuple[List[str], List[dict], Dict[str, float]]]:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if dataset_name not in payload:
        return None
    entry = payload[dataset_name]
    ranked = list(entry.get("benchmark", []))
    if top_s > len(ranked):
        raise ValueError(
            f"The benchmark cache stores {len(ranked)} datasets for {dataset_name}, but --s/--top_s={top_s}."
        )
    top_benchmarks = ranked[:top_s]
    suggestions = []
    performances = {}
    for benchmark_name in top_benchmarks:
        if benchmark_name not in entry:
            raise KeyError(f"Missing cached IMS design for {dataset_name} <- {benchmark_name}.")
        design = entry[benchmark_name]
        suggestions.append({dataset_name: {"link": design["link"], "ops": design["ops"]}})
        if "performance" in design:
            performances[benchmark_name] = float(design["performance"])
    return top_benchmarks, suggestions, performances


def build_metrics(args) -> List[str]:
    if args.no_statistics:
        return []
    if args.metrics:
        return list(args.metrics)
    return ALL_METRICS[: args.g]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="Planetoid:Cora")
    parser.add_argument("--unseen_dataset", dest="dataset", type=str, help="Alias for --dataset.")
    parser.add_argument("--user_description_path", type=str, default=None)
    parser.add_argument("--initialization", type=str, default="transfer", choices=["transfer", "naive", "none"])
    parser.add_argument("--search_strategy", type=str, default="llm_driven")

    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--g", type=int, default=8)
    parser.add_argument("--metrics", type=str, nargs="*", default=None)
    parser.add_argument("--s", "--top_s", dest="s", type=int, default=3)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--n_children", "--num_children", dest="n_children", type=int, default=30)

    parser.add_argument("--use_bad_designs", action="store_true", default=False)
    parser.add_argument("--use_semantic", action="store_true", default=False)
    parser.add_argument("--no_statistics", action="store_true", default=False)
    parser.add_argument("--no_reorder", action="store_true", default=False)
    parser.add_argument("--llm_no_candidates", action="store_true", default=False)
    parser.add_argument("--use_parser", action="store_true", default=False)
    parser.add_argument("--add_statistics", action="store_true", default=False)

    parser.add_argument("--benchmarking", action="store_true", default=False)
    parser.add_argument("--no_benchmark_cache", action="store_true", default=False)
    parser.add_argument("--dcm_only", action="store_true", default=False)
    parser.add_argument("--force_benchmark", type=str, default=None)

    parser.add_argument("--dcm_method", choices=["llm", "formula"], default="llm")
    parser.add_argument("--n_f", type=int, default=None)
    parser.add_argument(
        "--formula_uniform_weights",
        action="store_true",
        default=False,
        help="Use uniform w_u^k in formula DCM. By default formula DCM asks the LLM to induce w_u^k.",
    )

    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_model", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "responses"))
    parser.add_argument(
        "--benchmark_similarity_cache",
        type=str,
        default=str(REPO_ROOT / "LLMConfiguredAutoGNN" / "initial_model_benchmark.json"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_global_seed(args.seed)
    if args.search_strategy != "llm_driven":
        raise ValueError("The public release keeps the paper method path: --search_strategy llm_driven.")

    unseen_dataset_name = args.dataset
    short_name = canonical_name(unseen_dataset_name)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(args.output_dir)
    response_save_path = output_root / short_name / f"{short_name}_{args.search_strategy}_{current_time}"
    response_save_path.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    unseen_dir = as_source_root(data_root / "Unseen datasets")
    benchmark_dir = as_source_root(data_root / "Benchmark datasets")
    predefined_path = str(data_root / "Benchmark datasets" / "predefined_descriptions.json")

    benchmark_datasets = [dataset for dataset in DEFAULT_BENCHMARKS if canonical_name(dataset) != short_name]
    metrics_list = build_metrics(args)
    benchmarking = unseen_dataset_name in DEFAULT_BENCHMARKS and args.benchmarking

    user_description = read_text(args.user_description_path) or read_default_user_description(unseen_dir, unseen_dataset_name)

    llm = None
    latency1 = 0
    latency2 = 0
    latency3 = 0
    latency4 = 0

    def require_llm(model: Optional[str] = None):
        nonlocal llm
        if llm is None:
            llm = get_llm(args.openai_api_key, model or args.openai_model)
        if llm is None:
            raise RuntimeError("An OpenAI API key is required. Set OPENAI_API_KEY or pass --openai_api_key.")
        return llm

    t0 = time.time()
    understanding_module = GraphDatasetUnderstanding(
        unseen_dataset_name,
        user_description=user_description,
        metrics_list=metrics_list,
        root_dir=unseen_dir,
        no_statistics=args.no_statistics,
        use_semantic=args.use_semantic,
        num_samples=20,
        num_hops=2,
        seed=args.seed,
        predefined_descriptions_path=predefined_path,
    )
    unseen_dataset_description = understanding_module.process()

    most_similar = None
    similarity_scores = None
    suggested_design_dict_list = []
    cached_initial_performances = None
    llm_configured_autognn = None

    cache_result = None
    if benchmarking and args.initialization == "transfer" and not args.no_benchmark_cache:
        cache_result = load_benchmark_cache(Path(args.benchmark_similarity_cache), unseen_dataset_name, args.s)

    if cache_result is not None:
        top_benchmarks, suggested_design_dict_list, cached_initial_performances = cache_result
        most_similar = {unseen_dataset_name: top_benchmarks}
        print(f"most_similar: {most_similar}")
        print(f"suggested_design_dict: {suggested_design_dict_list}\n")
    elif args.initialization == "transfer":
        if args.force_benchmark:
            benchmark_name = list(BENCHMARK_MAPPING.keys())
            if args.force_benchmark not in benchmark_name:
                raise ValueError("Forced benchmark dataset not found in the benchmark datasets.")
            most_similar = {unseen_dataset_name: [args.force_benchmark]}
            print(f"most_similar: {most_similar}")
        else:
            benchmark_dataset_descriptions: Dict[str, str] = {}
            for benchmark_dataset_name in benchmark_datasets:
                benchmark_dataset_understanding = GraphDatasetUnderstanding(
                    benchmark_dataset_name,
                    user_description=None,
                    metrics_list=metrics_list,
                    root_dir=benchmark_dir,
                    no_statistics=args.no_statistics,
                    use_semantic=args.use_semantic,
                    num_samples=20,
                    num_hops=2,
                    seed=args.seed,
                    predefined_descriptions_path=predefined_path,
                )
                benchmark_dataset_descriptions[benchmark_dataset_name] = benchmark_dataset_understanding.process()
            latency1 = time.time() - t0
            print(f"Graph Dataset Understanding Latency: {latency1}")

            t0 = time.time()
            n_rank = max(args.s, args.n)
            task_similarity_file_name = response_save_path / f"1_task_similarity_response_{current_time}.txt"
            if args.dcm_method == "llm":
                dataset_comparison = GraphDatasetComparison(
                    unseen_dataset_description=unseen_dataset_description,
                    benchmark_dataset_descriptions=benchmark_dataset_descriptions,
                    unseen_dataset_name=unseen_dataset_name,
                    benchmark_datasets=benchmark_datasets,
                    unseen_dir=unseen_dir,
                    benchmark_dir=benchmark_dir,
                    use_parser=True,
                )
                textual_similarity_response = dataset_comparison.compare_datasets(
                    require_llm(args.openai_model),
                    str(task_similarity_file_name),
                )
                most_similar, similarity_scores = dataset_comparison.analyze_similarity_scores_from_dict(
                    textual_similarity_response,
                    unseen_dataset_name,
                    n_rank,
                )
            else:
                from GraphDatasetComparison.FormulaSimilarity import FormulaDatasetSimilarity, FormulaSimilarityConfig

                formula = FormulaDatasetSimilarity(
                    confidence_path=str(REPO_ROOT / "GraphDatasetComparison" / "property_confidence.json"),
                    cfg=FormulaSimilarityConfig(n_f=args.n_f or args.g),
                )
                unseen_properties = formula.load_properties_from_description(unseen_dataset_name, unseen_dir)
                benchmark_properties = {
                    canonical_name(dataset): formula.load_properties_from_description(dataset, benchmark_dir)
                    for dataset in benchmark_datasets
                }
                if args.formula_uniform_weights:
                    weights = formula.uniform_weights()
                    raw_weight_response = None
                else:
                    weights, raw_weight_response = formula.induce_weights(
                        require_llm(args.openai_model),
                        unseen_dataset_description,
                        {canonical_name(k): v for k, v in benchmark_dataset_descriptions.items()},
                    )
                flat_scores = formula.compute_scores(unseen_properties, benchmark_properties, weights)
                top_benchmarks = formula.rank(flat_scores, n_rank)
                most_similar = {unseen_dataset_name: top_benchmarks}
                similarity_scores = {unseen_dataset_name: flat_scores}
                task_similarity_file_name.write_text(
                    json.dumps(
                        {
                            "method": "formula",
                            "weights": weights,
                            "raw_weight_response": raw_weight_response,
                            "similarity_scores": flat_scores,
                            "top_datasets": top_benchmarks,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            latency2 = time.time() - t0
            print(f"Graph Dataset Comparison Latency: {latency2}")
            print(f"similarity_scores: {similarity_scores}")

        if args.dcm_only:
            return

        t0 = time.time()
        mode = "both" if args.use_bad_designs else "best"
        llm_configured_autognn = LLMConfiguredAutoGNN(most_similar, use_parser=args.use_parser, k=args.k, mode=mode)
        candidate_pools = llm_configured_autognn.generate_candidate_pools()
        models_info = candidate_pools[unseen_dataset_name]
        for i in range(args.s):
            file_path = response_save_path / f"2_{args.k}_suggested_design_response_{i}_{current_time}.txt"
            suggested_design_dict = llm_configured_autognn.suggest_initial_trial(
                unseen_dataset_name,
                [models_info[i]],
                langchain_query=require_llm(args.openai_model),
                similarities=similarity_scores,
                file_path=str(file_path),
            )
            suggested_design_dict_list.append(suggested_design_dict)
        print(f"suggested_design_dict: {suggested_design_dict_list}\n")
        latency3 = time.time() - t0
        print(f"Suggest Initial Trial Latency: {latency3}")
    else:
        if args.dcm_only:
            return
        t0 = time.time()
        llm_configured_autognn = LLMConfiguredAutoGNN(None, use_parser=args.use_parser, k=0, add_statistics=args.add_statistics)
        file_path = response_save_path / f"2_simple_suggested_design_response_{current_time}.txt"
        suggested_design_dict = llm_configured_autognn.suggest_initial_trial(
            unseen_dataset_name,
            None,
            langchain_query=require_llm("gpt-4-turbo"),
            description=unseen_dataset_description,
            file_path=str(file_path),
        )
        suggested_design_dict_list = [suggested_design_dict]
        print(f"suggested_design_dict: {suggested_design_dict_list[0]}\n")
        latency3 = time.time() - t0
        print(f"Suggest Initial Trial Latency: {latency3}")

    if args.dcm_only:
        return

    if llm_configured_autognn is None:
        mode = "both" if args.use_bad_designs else "best"
        llm_configured_autognn = LLMConfiguredAutoGNN(most_similar, use_parser=args.use_parser, k=args.k, mode=mode)
        llm_configured_autognn.generate_candidate_pools()

    t0 = time.time()
    initial_detailed_infos_list = []
    if benchmarking:
        data = None
        for suggested_design_dict in suggested_design_dict_list:
            initial_detailed_infos = llm_configured_autognn.extract_benchmark_results(
                unseen_dataset_name,
                suggested_design_dict,
                log=True,
            )
            initial_detailed_infos_list.append(initial_detailed_infos)
    else:
        unseen_dataset_reader = DatasetReader(unseen_dataset_name, unseen_dir)
        data = unseen_dataset_reader.read_dataset()
        for suggested_design_dict in suggested_design_dict_list:
            initial_detailed_infos = llm_configured_autognn.run_gnn_experiment(
                unseen_dataset_name,
                data,
                suggested_design_dict,
            )
            initial_detailed_infos_list.append(initial_detailed_infos)

    best_perf_dict = max(initial_detailed_infos_list, key=lambda x: x["perf"])
    print("LLM-suggested Initial Trial:")
    print(f"- Architecture: {best_perf_dict['link']}, Operations: {best_perf_dict['ops']}")
    print(f"- Performance: {best_perf_dict['perf']}\n")

    file_path = response_save_path / f"{args.search_strategy}_{current_time}.txt"
    if llm is None:
        llm = require_llm("gpt-3.5-turbo")
    best_detailed_infos, gnas_history = llm_configured_autognn.run_gnas_pipeline(
        unseen_dataset_name,
        data,
        initial_detailed_infos_list,
        max_iter=args.max_iter,
        n=args.n,
        langchain_query=llm,
        search_strategy=args.search_strategy,
        file_path=str(file_path),
        num_children=args.n_children,
        benchmarking=benchmarking,
        no_reorder=args.no_reorder,
        llm_no_candidates=args.llm_no_candidates,
    )
    print("LLM-suggested best Trial:")
    print(f"- Architecture: {best_detailed_infos['link']}, Operations: {best_detailed_infos['ops']}")
    print(f"- Performance: {best_detailed_infos['perf']}")

    performance_list = [
        max(entry["perf"] for entry in gnas_history[iter_num])
        if isinstance(gnas_history[iter_num], list)
        else gnas_history[iter_num]["perf"]
        for iter_num in gnas_history
    ]
    best_list = [
        max(entry["best"] for entry in gnas_history[iter_num])
        if isinstance(gnas_history[iter_num], list)
        else gnas_history[iter_num]["best"]
        for iter_num in gnas_history
    ]
    promoted_list = [
        gnas_history[iter_num]["promoted"]["perf"]
        if isinstance(gnas_history[iter_num], dict) and gnas_history[iter_num]["promoted"] is not None
        else 0
        for iter_num in gnas_history
    ]
    print("LLM-driven AutoGNN History (Real):")
    print(performance_list)
    print("LLM-driven AutoGNN History (Best):")
    print(best_list)
    print("LLM-driven AutoGNN History (Promoted):")
    print(promoted_list)
    latency4 = time.time() - t0
    print(f"LLM-driven AutoGNN Latency: {latency4}")

    summary_path = response_save_path / f"gnas_result_summary_{current_time}.txt"
    summary_path.write_text(
        "LLM-suggested best Trial:\n"
        f"- Architecture: {best_detailed_infos['link']}, Operations: {best_detailed_infos['ops']}\n"
        f"- Performance: {best_detailed_infos['perf']}\n"
        "LLM-driven AutoGNN History:\n"
        f"gnas_history: {gnas_history}\n\n"
        "Latencies:\n"
        f"Graph Dataset Understanding Latency: {latency1}\n"
        f"Graph Dataset Comparison Latency: {latency2}\n"
        f"Suggest Initial Trial Latency: {latency3}\n"
        f"LLM-driven AutoGNN Latency: {latency4}\n"
        f"Cached initial performances: {cached_initial_performances}\n",
        encoding="utf-8",
    )

    best_path = response_save_path / f"best_performances_{current_time}.txt"
    with open(best_path, "w", encoding="utf-8") as file:
        for best in performance_list:
            file.write(f"{best}, ")
        file.write("\n")
        for best in best_list:
            file.write(f"{best}, ")
        file.write("\n")
        for best in promoted_list:
            file.write(f"{best}, ")
        file.write("\n")

    plot_path = response_save_path / f"performance_vs_iteration_{current_time}.png"
    llm_configured_autognn.plot_performance_vs_iteration(
        gnas_history,
        str(plot_path),
        args.dataset,
        args.search_strategy,
    )


if __name__ == "__main__":
    main()



