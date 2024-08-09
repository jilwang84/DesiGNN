# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

from .KnowledgeQuery import KnowledgeQuery
from .LLMPromptConfigurator import LLMPromptConfigurator
from .NASBenchGraphGNN import run_gnn_experiment
from .GNASPipeline import GNASPipeline
import re
import matplotlib.pyplot as plt
from langchain_core.utils.function_calling import convert_to_openai_function


class LLMConfiguredAutoGNN:
    def __init__(self, similar_datasets, use_parser, k=20, mode='best', add_statistics=False):
        """
        Initializes the LLM Configured AutoGNN process.
        :param similar_datasets: Dictionary of dataset names similar to the unseen dataset.
        :param k: Number of top models to fetch for each selected dataset.
        """
        self.similar_datasets = similar_datasets
        self.gnn_benchmark = KnowledgeQuery(k, mode)
        self.prompter = LLMPromptConfigurator()
        self.candidate_pools = None
        self.gnas_pipeline = None
        self.use_parser = use_parser
        self.add_statistics = add_statistics

    def generate_candidate_pools(self):
        """
        Generates a candidate pool of model designs based on similar datasets.
        :return: Candidate pool of model designs.
        """
        self.candidate_pools = self.gnn_benchmark.get_models_for_source_datasets(self.similar_datasets)
        return self.candidate_pools

    def suggest_initial_trial(self, dataset_name, models_info, langchain_query=None, similarities=None,
                              description=None, file_path=None):
        suggested_design = self.query_llm_for_design_suggestion(dataset_name, models_info, langchain_query,
                                                                similarities, description)

        if self.use_parser:
            self.prompter.write_design_report(suggested_design, file_path)
            return self.prompter.reformat_suggested_design(suggested_design, dataset_name)
        else:
            with open(file_path, 'w') as file:
                file.write(suggested_design.content + "\n")
            return self.extract_model_designs(suggested_design.content, dataset_name)

    def run_gnas_pipeline(self, dataset_name, dataloader, initial_detailed_infos_list, max_iter, n, langchain_query,
                          search_strategy, file_path, num_children, benchmarking=False, no_reorder=False,
                          llm_no_candidates=False):
        if not no_reorder and search_strategy == 'designn' and len(self.similar_datasets[dataset_name]) == len(initial_detailed_infos_list):
            initial_detailed_infos_list = self.reorder_knowledge(self.similar_datasets, initial_detailed_infos_list, self.candidate_pools)
            print("Reordered knowledge based on similar datasets.")
        
        self.gnas_pipeline = GNASPipeline(search_strategy=search_strategy,
                                          llm_prompt_configurator=self.prompter,
                                          gnn_benchmark=self.gnn_benchmark,
                                          langchain_query=langchain_query,
                                          file_path=file_path,
                                          use_parser=self.use_parser,
                                          candidate_pools=self.candidate_pools,
                                          max_iter=max_iter,
                                          n=n,
                                          num_children=num_children)
        best_detailed_infos, gnas_history = self.gnas_pipeline.run_gnas(dataset_name, dataloader,
                                                                        initial_detailed_infos_list,
                                                                        benchmarking=benchmarking,
                                                                        llm_no_candidates=llm_no_candidates)

        return best_detailed_infos, gnas_history

    def query_llm_for_design_suggestion(self, dataset_name, models_info, langchain_query=None, similarities=None,
                                        description=None):
        """
        Queries the LLM for a model design suggestion for the unseen dataset.
        :param dataset_name: The source dataset.
        :param models_info: The top model designs.
        :param langchain_query: Function to query the LLM.
        :param similarities: Optional dictionary of similarity scores.
        :return: Suggested model design from the LLM.
        """
        if self.use_parser:
            if models_info:
                prompt, user_input, tool = self.prompter.generate_design_suggestion_prompt_parser(dataset_name,
                                                                                                  models_info, similarities)
            else:
                prompt, user_input, tool = self.prompter.generate_simple_design_suggestion_prompt_parser(dataset_name, 
                                                                                                         description)
            chain = prompt | langchain_query.with_structured_output(convert_to_openai_function(tool))

            response = chain.invoke({"input": user_input})

        else:
            prompt = self.prompter.generate_design_suggestion_prompt(dataset_name, models_info, similarities, 
                                                                     description)
            response = langchain_query.invoke(prompt)

        return response

    @staticmethod
    def run_gnn_experiment(dataset_name, dataloader, suggested_design_dict):
        detailed_infos = run_gnn_experiment(dataset_name, dataloader,
                                            suggested_design_dict[dataset_name]["link"],
                                            suggested_design_dict[dataset_name]["ops"])
        return detailed_infos
    
    def extract_benchmark_results(self, dataset_name, suggested_design_dict, log=False):
        detailed_infos = self.gnn_benchmark.extract_single_performance(dataset_name, suggested_design_dict)
        if log:
            detailed_infos['detailed_log'] = self.gnn_benchmark.extract_single_log(dataset_name, suggested_design_dict)
        return detailed_infos
    
    def reorder_knowledge(self, similar_datasets, initial_detailed_infos_list, candidate_pools):
        for key, datasets in similar_datasets.items():
            # Step 1: Extract performances and pair with full dataset info
            perf_info_pairs = [(info, dataset) for info, dataset in zip(initial_detailed_infos_list, datasets)]

            # Step 2: Sort perf_info_pairs based on 'perf' key, descending
            perf_info_pairs.sort(reverse=True, key=lambda x: x[0]['perf'])

            # Step 3: Update similar_datasets with sorted dataset names
            self.similar_datasets[key] = [dataset for _, dataset in perf_info_pairs]
            
            # Step 4: Update initial_detailed_infos_list to reflect new order
            initial_detailed_infos_list = [info for info, _ in perf_info_pairs]
            
            # Step 5: Reorder candidate_pools based on the sorted datasets
            dataset_to_pool = {pool['selected_dataset']: pool for pool in candidate_pools[key]}
            self.candidate_pools[key] = [dataset_to_pool[dataset] for _, dataset in perf_info_pairs]
 
        # Optionally return the updated structures
        return initial_detailed_infos_list

    @staticmethod
    def extract_model_designs(llm_response, dataset_name):
        """
        Extracts model designs suggested by the LLM for each source dataset.

        :param llm_response: A string containing the LLM's response in specified formats.
        :param dataset_name: The name of the source dataset used in the query.
        :return: A dictionary with source dataset names as keys and their suggested model designs as values.
        """
        # Patterns to match both formats of the LLM's response
        patterns = [
            r"\(Architecture: (\[.*?\]), Operations: (\[.*?\])\)",
            r"-\s*\*\*Architecture:\*\*\s*(\[.*?\])\s*-\s*\*\*Operations:\*\*\s*(\[.*?\])",
            r"-\s*\*\*Architecture\*\*:\s*(\[.*?\])\s*-\s*\*\*Operations\*\*:\s*(\[.*?\])",
            r"-\s*\*\*Architecture:\s*(\[.*?\])\*\*\s*-\s*\*\*Operations:\s*(\[.*?\])\*\*"
        ]

        # Initialize a dictionary to hold the extracted designs
        suggested_designs = {}

        for pattern in patterns:
            # Find all matches in the response
            matches = re.findall(pattern, llm_response)
            # Iterate through all matches and populate the dictionary
            for match in matches:
                architecture, operations = match
                # Evaluate to convert string representations to actual lists
                architecture_eval = eval(architecture)
                operations_eval = eval(operations)
                # Populate the dictionary with the specific source dataset and extracted values
                suggested_designs[dataset_name] = {
                    "link": architecture_eval,
                    "ops": operations_eval
                }
                return suggested_designs

        return suggested_designs

    def plot_performance_vs_iteration(self, gnas_history, file_path, dataset=None, search_strategy=None):
        # Extracting iteration numbers and their corresponding performance values
        iterations = list(gnas_history.keys())
        performances = [max(entry['perf'] for entry in gnas_history[iter_num]) if isinstance(gnas_history[iter_num], list) else gnas_history[iter_num]['perf'] for iter_num in gnas_history]

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, performances, marker='o', linestyle='-')
        if dataset and search_strategy:
            plt.title(f"{self.format_string_advanced(search_strategy)} on {self.extract_dataset_name(dataset)}")
        else:
            plt.title('Performance vs Iteration for NAS History')
        plt.xlabel('Iteration')
        plt.ylabel('Performance')
        plt.grid(True)
        plt.savefig(file_path)

    def plot_performance_vs_generation(self, gnas_history, file_path, dataset=None, search_strategy=None):
        # Initialize lists for data collection
        best_performances = []
        all_performances = []

        # Extract best performance per generation and all performances
        for generation, children in gnas_history.items():
            generation_performances = [child['perf'] for child in children]
            all_performances.extend([(int(generation), perf) for perf in generation_performances])
            best_performance = max(generation_performances)
            best_performances.append((int(generation), best_performance))

        # Sort based on generation for plotting consistency
        best_performances.sort()
        all_performances.sort()

        # Unpack the lists of tuples
        gen_nums, best_perf_vals = zip(*best_performances)
        all_gen_nums, all_perf_vals = zip(*all_performances)

        # Plotting all performances as scatter
        plt.scatter(all_gen_nums, all_perf_vals, color='lightgray', label='All Children')

        # Plotting best performances as line
        plt.plot(gen_nums, best_perf_vals, marker='o', linestyle='-', color='blue', label='Best Child')

        if dataset and search_strategy:
            plt.title(f"{self.format_string_advanced(search_strategy)} on {self.extract_dataset_name(dataset)}")
        else:
            plt.title('Best Child Performance vs Generation for NAS History')
        plt.xlabel('Generation')
        plt.ylabel('Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(file_path)

    @staticmethod
    def format_string_advanced(input_string):
        # Split the string by underscores
        words = input_string.split('_')
        # Capitalize each word appropriately; special case for 'llm'
        formatted_words = [word.upper() if word == 'llm' else word.capitalize() for word in words]
        # Join the words with a space and add a period at the end
        formatted_string = ' '.join(formatted_words) + '.'
        return formatted_string

    @staticmethod
    def extract_dataset_name(full_name):
        # Split the string by the colon
        parts = full_name.split(':')
        # Return the part after the colon if it exists, otherwise return the original input
        return parts[1] if len(parts) > 1 else full_name

