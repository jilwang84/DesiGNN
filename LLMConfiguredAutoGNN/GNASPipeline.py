# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

import re
import random
from langchain_core.utils.function_calling import convert_to_openai_function
from .NASBenchGraphGNN import run_gnn_experiment
from nas_bench_graph.architecture import Arch


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

        if 'pool' in self.search_strategy:
            self.candidate_pools = None
        if 'log' in self.search_strategy:
            self.use_training_log = False
        self.benchmarking = False

    def run_gnas(self, dataset_name, dataloader, initial_detailed_infos_list, benchmarking=False, 
                 llm_no_candidates=False):
        self.benchmarking = benchmarking
        if benchmarking and not any(info.get('detailed_log') for info in initial_detailed_infos_list):
            self.use_training_log = False
        if self.search_strategy == 'traditional':
            return self.traditional_gnas(dataset_name, dataloader, initial_detailed_infos_list[0])
        elif 'llm_refinement' in self.search_strategy:
            return self.llm_refinement(dataset_name, dataloader, initial_detailed_infos_list)
        elif 'llm_evolutionary' in self.search_strategy:
            return self.llm_evolutionary_search(dataset_name, dataloader, initial_detailed_infos_list[0])
        elif 'fast_evolution_with_llm' in self.search_strategy:
            return self.llm_evolutionary_search_with_fast_selection(dataset_name, dataloader, initial_detailed_infos_list[0])
        elif 'llm_driven' in self.search_strategy:
            return self.llm_driven_search(dataset_name, dataloader, initial_detailed_infos_list, llm_no_candidates)
        elif 'GPT4GNAS' in self.search_strategy:
            return self.GPT4GNAS_search(dataset_name, dataloader)
        elif 'kg' in self.search_strategy:
            return self.kg_search(dataset_name, dataloader, initial_detailed_infos_list)
        else:
            raise ValueError("Unsupported strategy specified.")
        

    def llm_driven_search(self, dataset_name, dataloader, initial_detailed_infos_list, llm_no_candidates):
        """
        Perform an LLM-based evolutionary search using Graph Neural Architecture search strategies.

        :param dataset_name: Name of the dataset being tested.
        :param dataloader: DataLoader providing the dataset for training and validation.
        :param initial_detailed_infos: Dictionary containing initial model design and its performance.
        :param num_children: Number of child models to generate each generation.
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
        for similar_dataset in list(self.candidate_pools.values())[0][1:n_initial]:     # 1:3 0:3
            merged_pool.extend(similar_dataset['top_models'])
        
        # Evolutionary search through generations
        top1_knowledge = self.candidate_pools[dataset_name][0]['selected_dataset']
        last_promoted = None
        for generation in range(self.max_iter):                    # Consider ten generations for this example
            promoted_child = None
            for _attempt in range(10):
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
            if promoted_child is None:
                promoted_child = children[0]
            
            promoted_child_performance = None
            if self.benchmarking:
                details = self.gnn_benchmark.extract_single_performance(dataset_name, {dataset_name: promoted_child})
                promoted_child_performance = details['perf']
                print(f"Generation {generation + 1}: Promoted child: {promoted_child['link']} {promoted_child['ops']} Performance: {promoted_child_performance}")

            # Construct prompt to let LLM select the most promising child
            if self.use_parser:
                raise NotImplementedError("Parser not supported for this search strategy.")
                prompt, user_input, optimization_tool = self.llm_prompt_configurator.generate_llm_selection_prompt_parser(
                    dataset_name, children, current_design, generation + 1, gnas_history, best_design,
                    self.use_training_log, self.candidate_pools)
                selected_child = self.query_llm_for_best_child_parser(prompt, user_input, optimization_tool,
                                                                      generation + 1, dataset_name)
            else:
                knowledge = self.candidate_pools if not llm_no_candidates else None
                prompt = self.llm_prompt_configurator.generate_llm_mutation_prompt(dataset_name, promoted_child, 
                                                                                   current_design, generation + 1, gnas_history, best_design, 
                                                                                   self.use_training_log,
                                                                                   knowledge)
                refined_child = self.query_llm_for_best_child(prompt, generation + 1, dataset_name)

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
        Generate a new child model by performing a multi-point crossover and blending operation between
        the current best design and a randomly selected model from the merged pool.

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

    def query_llm_for_design_refinement(self, prompt, iteration, dataset_name):
        refined_design = self.langchain_query.invoke(prompt)

        # Append the response to the file
        with open(self.file_path, 'a') as file:  # Open in append mode
            file.write(f"\nResponse for iteration {iteration}:\n")
            file.write(refined_design.content + "\n")
        refined_design_dict = self.llm_prompt_configurator.extract_model_designs(refined_design.content, dataset_name)

        return refined_design_dict

    def query_llm_for_design_evolution(self, prompt, generation):
        refined_design = self.langchain_query.invoke(prompt)

        # Append the response to the file
        with open(self.file_path, 'a') as file:  # Open in append mode
            file.write(f"\nResponse for generation {generation}:\n")
            file.write(refined_design.content + "\n")
        children = self.llm_prompt_configurator.extract_model_designs_evolution(refined_design.content)

        return children

    def query_llm_for_best_child(self, prompt, generation, dataset_name):
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

    def query_llm_for_design_refinement_parser(self, prompt, user_input, optimization_tool, iteration, dataset_name):
        chain = prompt | self.langchain_query.with_structured_output(convert_to_openai_function(optimization_tool))
        response = chain.invoke({"input": user_input})

        self.llm_prompt_configurator.write_optimization_report(response, self.file_path, iteration)

        return self.llm_prompt_configurator.reformat_refined_design(response, dataset_name)

    def query_llm_for_design_evolution_parser(self, prompt, user_input, optimization_tool, generation, dataset_name,
                                              num_children):
        chain = prompt | self.langchain_query.with_structured_output(convert_to_openai_function(optimization_tool))
        response = chain.invoke({"input": user_input})

        self.llm_prompt_configurator.write_evolutionary_report(response, self.file_path, generation, dataset_name,
                                                               num_children)

        return self.llm_prompt_configurator.reformat_evolutionary_design(response, dataset_name, num_children)

    def query_llm_for_best_child_parser(self, prompt, user_input, optimization_tool, generation, dataset_name):
        chain = prompt | self.langchain_query.with_structured_output(convert_to_openai_function(optimization_tool))
        response = chain.invoke({"input": user_input})

        self.llm_prompt_configurator.write_optimization_report(response, self.file_path, generation)

        return self.llm_prompt_configurator.reformat_refined_design(response, dataset_name)
    
    def GPT4GNAS_search(self, dataset_name, dataloader):
        """
        Run the Graph Neural Architecture Search (GNAS) pipeline.

        :param dataset_name: Name of the dataset being tested.
        :param dataloader: Dataset used for training and validation.
        :param initial_detailed_infos: Initial model design suggested by LLM.
        """
        best_performance = 0
        best_design = None
        gnas_history = {}
        for generation in range(int(self.max_iter/self.num_children)):
            print(f"Generation {generation + 1}:")
            #prompt = self.llm_prompt_configurator.generate_GPT4GNAS_prompt(self.num_children, generation, gnas_history)
            prompt = self.llm_prompt_configurator.generate_GHGNAS_prompt(dataset_name, self.num_children, generation, gnas_history)
            # prompt = self.llm_prompt_configurator.generate_design_evolution_prompt(dataset_name, self.num_children,
            #                                                                        generation, gnas_history, best_design, False,
            #                                                                        False)
            children = self.query_llm_for_design_evolution(prompt, generation + 1)

            # Evaluate children
            best_child = None
            best_child_performance = float('-inf')
            children_history = []
            for child in children:
                if self.benchmarking:
                    new_detailed_infos = self.gnn_benchmark.extract_single_performance(dataset_name, 
                                                                                       {dataset_name: child})
                else:
                    new_detailed_infos = run_gnn_experiment(dataset_name, dataloader, 
                                                            child["link"], 
                                                            child["ops"])

                children_history.append({
                    'perf': new_detailed_infos['perf'],
                    'link': new_detailed_infos['link'],
                    'ops': new_detailed_infos['ops']
                })

                performance = new_detailed_infos['perf']
                if performance > best_child_performance:
                    best_child = new_detailed_infos
                    best_child_performance = performance
                print(f" - Suggested new model design {new_detailed_infos['link']} {new_detailed_infos['ops']} "
                      f"Performance: {performance}")

            if best_child_performance > best_performance:
                best_design = best_child
                best_design['iteration'] = generation + 1
                best_performance = best_child_performance
            print(
                f"Generation {generation + 1}: Suggested best model design {best_child['link']}"
                f" {best_child['ops']} Performance: {best_child_performance}")
            
            for child in children_history:
                child['best'] = best_performance

            gnas_history[str(generation + 1)] = children_history

        return best_design, gnas_history

    def traditional_gnas(self, unseen_dataset_name, data, initial_design):
        # Placeholder for traditional NAS using a library like GraphNAS or AutoGL
        print("Running traditional GNAS...")
        # Implement specific traditional NAS logic here
        # This would typically involve setting up a search space, running the NAS algorithm, and selecting the best model
        return None

    def llm_refinement(self, dataset_name, dataloader, initial_detailed_infos_list):
        """
        Run the Graph Neural Architecture Search (GNAS) pipeline.

        :param dataset_name: Name of the dataset being tested.
        :param dataloader: Dataset used for training and validation.
        :param initial_detailed_infos: Initial model design suggested by LLM.
        """
        # current_design = initial_detailed_infos
        # best_performance = initial_detailed_infos['perf']
        # best_design = initial_detailed_infos
        # best_design['iteration'] = 0
        # gnas_history = {
        #     '0': {
        #         'perf': initial_detailed_infos['perf'],
        #         'link': initial_detailed_infos['link'],
        #         'ops': initial_detailed_infos['ops']
        #     }
        # }
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

        for generation in range(self.max_iter):                    # Consider ten generations for this example
            knowledge = self.candidate_pools
            prompt = self.llm_prompt_configurator.generate_llm_prompt(dataset_name,
                                                                      current_design,
                                                                      generation + 1,
                                                                      gnas_history,
                                                                      best_design, 
                                                                      self.use_training_log,
                                                                      knowledge)
            refined_child = self.query_llm_for_best_child(prompt, generation + 1, dataset_name)

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
                'best': best_performance
            }
            current_design = new_detailed_infos
            generation += 1

        return best_design, gnas_history

    def llm_evolutionary_search(self, dataset_name, dataloader, initial_detailed_infos, num_children=3):
        """
        Run the Graph Neural Architecture Search (GNAS) pipeline.

        :param dataset_name: Name of the dataset being tested.
        :param dataloader: Dataset used for training and validation.
        :param initial_detailed_infos: Initial model design suggested by LLM.
        """
        current_design = initial_detailed_infos
        best_performance = initial_detailed_infos['perf']
        best_design = initial_detailed_infos
        best_design['iteration'] = 0
        gnas_history = {
            '0': [{
                'perf': initial_detailed_infos['perf'],
                'link': initial_detailed_infos['link'],
                'ops': initial_detailed_infos['ops']
            }]
        }
        for generation in range(int(self.max_iter/self.num_children)):
            print(f"Generation {generation + 1}:")
            if self.use_parser:
                prompt, user_input, optimization_tool = self.llm_prompt_configurator.generate_design_evolution_prompt_parser(
                    dataset_name, self.num_children, current_design, generation + 1, gnas_history, best_design,
                    self.use_training_log, self.candidate_pools)
                children = self.query_llm_for_design_evolution_parser(prompt, user_input, optimization_tool,
                                                                      generation + 1, dataset_name, num_children)
            else:
                prompt = self.llm_prompt_configurator.generate_design_evolution_prompt(dataset_name, self.num_children,
                                                                                       current_design, generation + 1,
                                                                                       gnas_history, best_design,
                                                                                       self.use_training_log,
                                                                                       self.candidate_pools)
                print(prompt)
                children = self.query_llm_for_design_evolution(prompt, generation + 1)

            # Evaluate children
            best_child = None
            best_child_performance = float('-inf')
            children_history = []
            for child in children:
                if self.benchmarking:
                    new_detailed_infos = self.gnn_benchmark.extract_single_performance(dataset_name, 
                                                                                       {dataset_name: child})
                else:
                    new_detailed_infos = run_gnn_experiment(dataset_name, dataloader, 
                                                            child["link"], 
                                                            child["ops"])

                children_history.append({
                    'perf': new_detailed_infos['perf'],
                    'link': new_detailed_infos['link'],
                    'ops': new_detailed_infos['ops']
                })

                performance = new_detailed_infos['perf']
                if performance > best_child_performance:
                    best_child = new_detailed_infos
                    best_child_performance = performance
                print(f" - Suggested new model design {new_detailed_infos['link']} {new_detailed_infos['ops']} "
                      f"Performance: {performance}")

            if best_child_performance > best_performance:
                best_design = best_child
                best_design['iteration'] = generation + 1
                best_performance = best_child_performance
            print(
                f"Generation {generation + 1}: Suggested best model design {best_child['link']}"
                f" {best_child['ops']} Performance: {best_child_performance}")
            
            for child in children_history:
                child['best'] = best_performance

            gnas_history[str(generation + 1)] = children_history

        return best_design, gnas_history

    def llm_evolutionary_search_with_fast_selection(self, dataset_name, dataloader, initial_detailed_infos,
                                                    num_children=10):
        """
        Perform an LLM-based evolutionary search using Graph Neural Architecture search strategies.

        :param dataset_name: Name of the dataset being tested.
        :param dataloader: DataLoader providing the dataset for training and validation.
        :param initial_detailed_infos: Dictionary containing initial model design and its performance.
        :param num_children: Number of child models to generate each generation.
        """
        current_design = initial_detailed_infos
        best_performance = initial_detailed_infos['perf']
        best_design = initial_detailed_infos
        best_design['iteration'] = 0
        gnas_history = {
            '0': {
                'perf': initial_detailed_infos['perf'],
                'link': initial_detailed_infos['link'],
                'ops': initial_detailed_infos['ops']
            }
        }

        merged_pool = []
        for similar_dataset in list(self.candidate_pools.values())[0][1:3]:     # 1:3 0:3
            merged_pool.extend(similar_dataset['top_models'])
        # Evolutionary search through generations
        for generation in range(10):                    # Consider ten generations for this example
            children = []
            # Exploration: Generate new models using mutation and crossover from candidate pools
            for _ in range(num_children):
                child = self.generate_child(best_design, merged_pool)
                children.append(child)

            # Construct prompt to let LLM select the most promising child
            if self.use_parser:
                prompt, user_input, optimization_tool = self.llm_prompt_configurator.generate_llm_selection_prompt_parser(
                    dataset_name, children, current_design, generation + 1, gnas_history, best_design,
                    self.use_training_log, self.candidate_pools)
                selected_child = self.query_llm_for_best_child_parser(prompt, user_input, optimization_tool,
                                                                      generation + 1, dataset_name)
            else:
                prompt = self.llm_prompt_configurator.generate_llm_selection_prompt(dataset_name, children,
                                                                                    current_design, generation + 1,
                                                                                    gnas_history, best_design,
                                                                                    self.use_training_log,
                                                                                    self.candidate_pools)
                selected_child = self.query_llm_for_best_child(prompt, generation + 1, dataset_name)

            # Evaluate the selected child using the model training and validation function
            new_detailed_infos = run_gnn_experiment(dataset_name, dataloader,
                                                    selected_child[dataset_name]["link"],
                                                    selected_child[dataset_name]["ops"])
            performance = new_detailed_infos['perf']
            if performance > best_performance:
                best_design = new_detailed_infos
                best_design['iteration'] = generation + 1
                best_performance = performance
            print(
                f"Generation {generation + 1}: Suggested new model design {selected_child[dataset_name]['link']}"
                f" {selected_child[dataset_name]['ops']} Performance: {performance}")

            # Update current design with the new suggested design
            gnas_history[str(generation + 1)] = {
                'perf': new_detailed_infos['perf'],
                'link': new_detailed_infos['link'],
                'ops': new_detailed_infos['ops']
            }
            current_design = new_detailed_infos
            generation += 1

        return best_design, gnas_history

    def generate_child(self, current_design, merged_pool):
        """
        Generate a new child model by performing a multi-point crossover and blending operation between
        the current best design and a randomly selected model from the merged pool.

        :param current_design: The current best design, typically from previous iterations.
        :param merged_pool: A list containing the top models from the two most similar datasets.
        :return: A dictionary representing the child model with new 'link' (architecture) and 'ops' (operations).
        """
        # Randomly select a model from the merged pool for crossover
        random_model = random.choice(merged_pool)

        # Determine two crossover points for more complex mixing
        points = sorted(random.sample(range(1, min(len(current_design['link']), len(random_model[0])) - 1), 2))

        # Perform multi-point crossover with segment-wise blending
        new_architecture = (current_design['link'][:points[0]] +
                            random_model[0][points[0]:points[1]] +
                            current_design['link'][points[1]:])
        new_operations = (current_design['ops'][:points[0]] +
                          random_model[1][points[0]:points[1]] +
                          current_design['ops'][points[1]:])

        # Further blending within segments
        # Randomly swap operations within segments to increase diversity
        #for segment_start, segment_end in zip([0] + points, points + [None]):  # iterate over each segment
        #    segment_length = segment_end - segment_start if segment_end else len(new_operations) - segment_start
        #    if segment_length > 1:
        #        swap_indices = random.sample(range(segment_start, segment_end if segment_end else len(new_operations)),
        #                                     2)
        #        new_operations[swap_indices[0]], new_operations[swap_indices[1]] = new_operations[swap_indices[1]], \
        #                                                                           new_operations[swap_indices[0]]

        return {'link': new_architecture, 'ops': new_operations}

    '''
    def generate_child(self, current_design, merged_pool):
        """
        Generate a new child model by performing crossover and mutation based on the current design and candidate pools.
        """
        # Select a random model from the merged pool for crossover
        random_model = random.choice(merged_pool)

        # Crossover
        crossover_point = random.randint(1, len(current_design['link']) - 1)
        new_architecture = current_design['link'][:crossover_point] + random_model[0][crossover_point:]
        new_operations = current_design['ops'][:crossover_point] + random_model[1][crossover_point:]

        # Mutation: Randomly mutate one position in the architecture and operations
        #mutation_idx = random.randint(0, len(new_architecture) - 1)
        #new_architecture[mutation_idx] = random.randint(0, max(new_architecture))
        #new_operations[mutation_idx] = random.choice(
        #    ['gat', 'gcn', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip'])

        return {'link': new_architecture, 'ops': new_operations}
    '''

    @staticmethod
    def crossover(parent1, parent2):
        """
        Perform a single-point crossover between two GNN architectures.

        :param parent1: A dictionary containing 'Architecture' and 'Operations' of the first parent.
        :param parent2: A dictionary containing 'Architecture' and 'Operations' of the second parent.
        :return: A new child architecture derived from the two parents.
        """
        # Get crossover point
        crossover_point = random.randint(1, min(len(parent1['Architecture']), len(parent2['Architecture'])) - 1)

        # Create new architecture and operations by combining parts of both parents
        new_architecture = parent1['Architecture'][:crossover_point] + parent2['Architecture'][crossover_point:]
        new_operations = parent1['Operations'][:crossover_point] + parent2['Operations'][crossover_point:]

        return {'Architecture': new_architecture, 'Operations': new_operations}

    @staticmethod
    def mutate(architecture, mutation_rate=0.1):
        """
        Mutate a GNN architecture by randomly altering its structure or operations.

        :param architecture: A dictionary containing 'Architecture' and 'Operations'.
        :param mutation_rate: Probability of each element being mutated.
        :return: A mutated architecture.
        """
        # Possible operations for mutation
        possible_operations = ['gcn', 'gat', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip']

        # Mutate architecture
        new_architecture = [
            node if random.random() > mutation_rate else random.choice(range(len(architecture['Architecture'])))
            for node in architecture['Architecture']
        ]

        # Mutate operations
        new_operations = [
            op if random.random() > mutation_rate else random.choice(possible_operations)
            for op in architecture['Operations']
        ]

        return {'Architecture': new_architecture, 'Operations': new_operations}
    
    def kg_search(self, dataset_name, dataloader, initial_detailed_infos_list):
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
                'best': best_performance
            })
        
        # Evolutionary search through generations
        from kgnasmaster.KGNAS import KGNAS
        import pandas as pd
        kgnas = KGNAS(numerical_weight=0.5)
        kgnas.standardize = False
        kgnas.activation = 'power'
        kgnas.power = 1/3
        kgnas.upper_bound = 0.8
        kgnas.process_method = 'normal'
        kgnas.bound_frac = 0.5
        kgnas.set_num_weight(1.0)
        benchmark_mapping = {
        "Cora" : "cora",
        "CiteSeer" : "citeseer",
        "PubMed": "pubmed",
        "CS" : "cs",
        "Physics" : "physics",
        "Photo" : "photo",
        "Computers" : "computers",
        "ogbn-arxiv" : "arxiv",
        "ogbn-proteins" : "proteins"
        }
        current_design_id = Arch(current_design['link'], current_design['ops']).valid_hash()
        for generation in range(self.max_iter):                    # Consider ten generations for this example
            knowledge = []
            similar_model_df = kgnas.get_similar_model(source_dataset=benchmark_mapping[self.candidate_pools[dataset_name][0]['selected_dataset']], source_model=current_design_id, top_k_dataset=3, top_k_model=20, sim_metric='l2', sim_weights=[1, 4])
            for _, row in similar_model_df.iterrows():
                link = row['has_struct_topology']
                ops = [row[f'has_struct_{i}'] for i in range(1, 5)]
                knowledge.append((link, ops))
            
            prompt = self.llm_prompt_configurator.generate_kg_prompt(current_design, 
                                                                     generation + 1, 
                                                                     gnas_history, 
                                                                     best_design, 
                                                                     self.use_training_log,
                                                                     knowledge)
            
            refined_child = self.query_llm_for_best_child(prompt, generation + 1, dataset_name)

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
                'best': best_performance
            }
            current_design = new_detailed_infos
            current_design_id = Arch(current_design['link'], current_design['ops']).valid_hash()
            generation += 1

        return best_design, gnas_history

    # def kg_search(self, dataset_name, dataloader, initial_detailed_infos_list):
    #     n_initial = len(initial_detailed_infos_list)
    #     if n_initial < self.n:
    #         raise ValueError(f"Number of initial designs ({n_initial}) is less than the required number of designs ({self.n}).")

    #     initial_detailed_infos = max(initial_detailed_infos_list, key=lambda x: x['perf'])
    #     current_design = initial_detailed_infos
    #     best_performance = initial_detailed_infos['perf']
    #     best_design = initial_detailed_infos
    #     best_design['iteration'] = 0
    #     gnas_history = {
    #         '0': []
    #     }
    #     for i in range(n_initial):
    #         gnas_history['0'].append({
    #             'perf': initial_detailed_infos_list[i]['perf'],
    #             'link': initial_detailed_infos_list[i]['link'],
    #             'ops': initial_detailed_infos_list[i]['ops'],
    #             'best': best_performance
    #         })
        
    #     # Evolutionary search through generations
    #     from kgnasmaster.KGNAS import KGNAS
    #     import pandas as pd
    #     kgnas = KGNAS(numerical_weight=0.5)
    #     benchmark_mapping = {
    #     "Planetoid:Cora" : "cora",
    #     "Planetoid:CiteSeer" : "citeseer",
    #     "Planetoid:PubMed": "pubmed",
    #     "Coauthor:CS" : "cs",
    #     "Coauthor:Physics" : "physics",
    #     "Amazon:Photo" : "photo",
    #     "Amazon:Computers" : "computers",
    #     "ogbn-arxiv" : "arxiv",
    #     "ogbn-proteins" : "proteins"
    #     }
    #     candidate_df = kgnas.recommend_model(benchmark_mapping[dataset_name], top_k_dataset=3, top_k_model=20, score_metric='avg', include_target_dataset=False)
    #     candidate_model = candidate_df.iloc[0].copy()
    #     for generation in range(self.max_iter):                    # Consider ten generations for this example
    #         candidate_model['has_struct_topology'] = str(current_design['link'])
    #         candidate_model['has_struct_1'] = current_design['ops'][0]
    #         candidate_model['has_struct_2'] = current_design['ops'][1]
    #         candidate_model['has_struct_3'] = current_design['ops'][2]
    #         candidate_model['has_struct_4'] = current_design['ops'][3]
            
    #         knowledge = []
    #         similar_model_df = kgnas.get_similar_model(candidate_model, candidate_df, topk=30, sim_metric='l2')
    #         for _, row in similar_model_df.iterrows():
    #             link = eval(row['has_struct_topology'])
    #             ops = [row[f'has_struct_{i}'] for i in range(1, 5)]
    #             knowledge.append((link, ops))
    #         prompt = self.llm_prompt_configurator.generate_kg_prompt(current_design, 
    #                                                                  generation + 1, 
    #                                                                  gnas_history, 
    #                                                                  best_design, 
    #                                                                  self.use_training_log,
    #                                                                  knowledge)
    #         refined_child = self.query_llm_for_best_child(prompt, generation + 1, dataset_name)

    #         # Evaluate the selected child using the model training and validation function
    #         if self.benchmarking:
    #             new_detailed_infos = self.gnn_benchmark.extract_single_performance(dataset_name, refined_child)
    #             new_detailed_infos['detailed_log'] = self.gnn_benchmark.extract_single_log(dataset_name, refined_child)
    #         else:
    #             new_detailed_infos = run_gnn_experiment(dataset_name, dataloader, refined_child[dataset_name]["link"],
    #                                                     refined_child[dataset_name]["ops"])
    #         performance = new_detailed_infos['perf']
    #         if performance > best_performance:
    #             best_design = new_detailed_infos
    #             best_design['iteration'] = generation + 1
    #             best_performance = performance
    #         print(f"Generation {generation + 1}: Suggested new model design {refined_child[dataset_name]['link']} {refined_child[dataset_name]['ops']} Performance: {performance}")
            
    #         # Update current design with the new suggested design
    #         gnas_history[str(generation + 1)] = {
    #             'perf': new_detailed_infos['perf'],
    #             'link': new_detailed_infos['link'],
    #             'ops': new_detailed_infos['ops'],
    #             'best': best_performance
    #         }
    #         current_design = new_detailed_infos
    #         generation += 1

    #     return best_design, gnas_history

