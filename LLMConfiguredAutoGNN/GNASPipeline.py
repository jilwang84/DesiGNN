# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

import random
from langchain_core.utils.function_calling import convert_to_openai_function
from sympy import N
from .NASBenchGraphGNN import run_gnn_experiment


class GNASPipeline:
    def __init__(self, search_strategy, llm_prompt_configurator, gnn_benchmark, langchain_query, file_path, use_parser,
                 candidate_pools=None, max_iter=10, n=1, num_children=1):
        """
        Initializes the Graph NAS process.
        :param search_strategy: Search strategy used in the Graph NAS process.
        :param candidate_pools: Candidate pool of model designs.
        """
        self.search_strategy = search_strategy
        self.langchain_query = langchain_query
        self.candidate_pools = candidate_pools
        self.use_training_log = True
        self.gnn_benchmark = gnn_benchmark
        self.llm_prompt_configurator = llm_prompt_configurator
        self.file_path = file_path
        self.use_parser = use_parser
        self.max_iter = max_iter
        self.n = n
        self.num_children = num_children
        self.benchmarking = False

    def run_gnas(self, dataset_name, dataloader, initial_detailed_infos_list, benchmarking=False, 
                 llm_no_candidates=False):
        self.benchmarking = benchmarking
        if 'designn' in self.search_strategy:
            return self.designn_search(dataset_name, dataloader, initial_detailed_infos_list, llm_no_candidates)
        else:
            raise ValueError("Unsupported strategy specified.")
        

    def designn_search(self, dataset_name, dataloader, initial_detailed_infos_list, llm_no_candidates):
        """
        Perform the Model Proposal Refinement of DesiGNN.

        :param dataset_name: Name of the dataset being tested.
        :param dataloader: DataLoader providing the dataset for training and validation.
        :param initial_detailed_infos: Dictionary containing initial model design and its performance.
        :param llm_no_candidates: Flag indicating whether the LLM uses knowledge to refine the design.
        """
        n_initial = len(initial_detailed_infos_list)
        if n_initial < self.n:
            raise ValueError(f"Number of initial designs ({n_initial}) is less than the required number of designs ({self.n}).")

        initial_detailed_infos = max(initial_detailed_infos_list, key=lambda x: x['perf'])
        current_design = initial_detailed_infos
        best_performance = initial_detailed_infos['perf']
        best_design = initial_detailed_infos
        best_design['iteration'] = 0
        gnas_history = {
            '0': []
        }
        for i in range(n_initial):
            gnas_history['0'].append({
                'perf': initial_detailed_infos_list[i]['perf'],
                'link': initial_detailed_infos_list[i]['link'],
                'ops': initial_detailed_infos_list[i]['ops'],
                'best': best_performance,
                'promoted': None
            })

        merged_pool = []
        for similar_dataset in list(self.candidate_pools.values())[0][1:n_initial]:
            merged_pool.extend(similar_dataset['top_models'])
        
        # Evolutionary search through generations
        top1_knowledge = self.candidate_pools[dataset_name][0]['selected_dataset']
        last_promoted = None
        for generation in range(self.max_iter):
            while True:
                children = []
                # Exploration: Generate new models using mutation and crossover from candidate pools
                for _ in range(self.num_children):
                    child = self.controlled_exploration(best_design, merged_pool)
                    children.append(child)
                estimated_performances = self.gnn_benchmark.extract_performances(top1_knowledge, children)
                promoted_child = children[estimated_performances.index(max(estimated_performances))]

                if last_promoted is None or promoted_child != last_promoted:
                    last_promoted = promoted_child
                    break
            
            promoted_child_performance = None
            if self.benchmarking:
                details = self.gnn_benchmark.extract_single_performance(dataset_name, {dataset_name: promoted_child})
                promoted_child_performance = details['perf']
                print(f"Generation {generation + 1}: Promoted child: {promoted_child['link']} {promoted_child['ops']} Performance: {promoted_child_performance}")

            # Construct prompt to let LLM select the most promising child
            if self.use_parser:
                raise NotImplementedError("Parser not supported for this search strategy.")
            else:
                knowledge = self.candidate_pools if not llm_no_candidates else None
                prompt = self.llm_prompt_configurator.generate_llm_mutation_prompt(dataset_name, promoted_child, 
                                                                                   current_design, generation + 1, gnas_history, best_design, 
                                                                                   self.use_training_log,
                                                                                   knowledge)
                refined_child = self.query_llm_for_directional_exploitation(prompt, generation + 1, dataset_name)

            # Evaluate the selected child using the model training and validation function
            if self.benchmarking:
                new_detailed_infos = self.gnn_benchmark.extract_single_performance(dataset_name, refined_child)
                new_detailed_infos['detailed_log'] = self.gnn_benchmark.extract_single_log(dataset_name, refined_child)
            else:
                new_detailed_infos = run_gnn_experiment(dataset_name, dataloader, refined_child[dataset_name]["link"],
                                                        refined_child[dataset_name]["ops"])
            performance = new_detailed_infos['perf']
            if performance > best_performance:
                best_design = new_detailed_infos
                best_design['iteration'] = generation + 1
                best_performance = performance
            print(f"Generation {generation + 1}: Suggested new model design {refined_child[dataset_name]['link']} {refined_child[dataset_name]['ops']} Performance: {performance}")

            # Update current design with the new suggested design
            gnas_history[str(generation + 1)] = {
                'perf': new_detailed_infos['perf'],
                'link': new_detailed_infos['link'],
                'ops': new_detailed_infos['ops'],
                'best': best_performance,
                'promoted': {
                    'link': promoted_child['link'],
                    'ops': promoted_child['ops'],
                    'perf': promoted_child_performance
                }
            }
            current_design = new_detailed_infos
            generation += 1

        return best_design, gnas_history

    def controlled_exploration(self, current_design, merged_pool):
        """
        Generate a new child model by performing a controlled crossover between the current best design 
        and a randomly selected model from the merged pool.

        :param current_design: The current best design, typically from previous iterations.
        :param merged_pool: A list containing the top models from the two most similar datasets.
        :return: A dictionary representing the child model with new 'link' (architecture) and 'ops' (operations).
        """
        # Randomly select a model from the merged pool for crossover
        random_model = random.choice(merged_pool)
        
        # 1. Perform single-point crossover with adaptive rolling.
        first_part_a, second_part_a = current_design['link'][:2], current_design['link'][2:]
        first_part_b, second_part_b = random_model[0][:2], random_model[0][2:]
        candidates = [[0, 0], [0, 1]]
        overlap = [sp for sp in candidates if sp == first_part_a or sp == first_part_b]
        first_part_child = random.choice(overlap)

        # Get possible second parts based on the selected first part
        candidates = self.second_part_candidates(first_part_child)
        overlap = [sp for sp in candidates if sp == second_part_a or sp == second_part_b]
        if overlap:
            second_part_child = random.choice(overlap)
        else:
            second_part_child = random.choice(candidates)

        # Combine first and second parts to form the child
        new_architecture = first_part_child + second_part_child

        # 2. Introduce slight changes to a promising operation list based on another example.
        differences = [i for i, (a, b) in enumerate(zip(current_design['ops'], random_model[1])) if a != b]
    
        # Decide on the number of changes; here we use 1 or 2 changes for 'slight' modification
        if new_architecture == current_design['link']:
            num_changes = random.choice([1, 2, 3]) if len(differences) > 1 else 1
        else:
            num_changes = random.choice([0, 1, 2, 3]) if len(differences) > 1 else 1
        
        # Select random differences to change
        change_points = random.sample(differences, min(num_changes, len(differences)))
        
        # Create a copy of the promising list to modify
        new_operations = current_design['ops'][:]
        
        # Introduce changes at the selected points
        for point in change_points:
            new_operations[point] = random_model[1][point]

        return {'link': new_architecture, 'ops': new_operations}
    
    @staticmethod
    def second_part_candidates(first_part_child):
        """ Return valid second parts based on the first part of the structure. """
        if first_part_child == [0, 0]:
            return [[0, 0], [0, 1], [1, 1], [1, 2], [1, 3]]
        elif first_part_child == [0, 1]:
            return [[1, 1], [1, 2], [2, 2], [2, 3]]
        return []  # Return an empty list if the first part is not recognized

    def query_llm_for_directional_exploitation(self, prompt, generation, dataset_name):
        try:
            refined_design = self.langchain_query.invoke(prompt, timeout=120)
        except TimeoutError:
            print(f"Timeout for generation {generation}.")
            refined_design = self.langchain_query.invoke(prompt, timeout=120)

        # Append the response to the file
        with open(self.file_path, 'a') as file:  # Open in append mode
            file.write(f"\nResponse for generation {generation}:\n")
            file.write(refined_design.content + "\n")
        children = self.llm_prompt_configurator.extract_model_designs(refined_design.content, dataset_name)

        return children

