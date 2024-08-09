# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

from ast import Not
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, create_model, conlist


class LLMPromptConfigurator:

    @staticmethod
    def generate_design_suggestion_prompt_parser(dataset_name, models_info, similarities=None):
        """
        Generates a prompt asking the LLM for a model design suggestion.
        :param dataset_name: The source dataset.
        :param models_info: The top model designs.
        :param similarities: Optional dictionary of similarity scores.
        :return: A formatted LLM prompt string.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a machine learning expert proficient in Graph Neural Networks (GNN) design and graph "
                        "dataset understanding. Your task is to recommend a GNN model architecture that performs well "
                        "on the unseen dataset based on the top-performing and bad-performing GNN model architectures on the most similar "
                        "benchmark dataset to the user.\n"
                        "In the context of GNN, the design of a model is described by two main components: the "
                        "operation list and the macro architecture list. Here are the detailed settings:\n"
                        "1. The operation list is a list of four strings. We consider 9 candidate operations, which "
                        "are:\n"
                        "- 'gat': Graph Attention Network layer, utilizing attention mechanisms to weigh the "
                        "importance of nodes' neighbors.\n"
                        "- 'gcn': Graph Convolutional Network layer, applying a convolutional operation over the "
                        "graph to aggregate neighborhood information.\n"
                        "- 'gin': Graph Isomorphism Network layer, designed to capture the graph structure in the "
                        "embedding.\n"
                        "- 'cheb': Chebyshev Spectral Graph Convolution, using Chebyshev polynomials to filter graph "
                        "signals.\n"
                        "- 'sage': GraphSAGE, sampling and aggregating features from a node's neighborhood.\n"
                        "- 'arma': ARMA layer, utilizing Auto-Regressive Moving Average filters for graph "
                        "convolution.\n"
                        "- 'graph': k-GNN, extending the GNN to capture k-order graph motifs.\n"
                        "- 'fc': Fully Connected layer, a dense layer that does not utilize graph structure.\n"
                        "- 'skip': Skip Connection, enabling the creation of residual connections.\n"
                        "For example, an operation list could be ['gcn', 'gin', 'fc', 'cheb'], with 'gcn' as the first "
                        "computing node. The order of operations in the list matters. \n"
                        "2. The macro architecture list is represented as a directed acyclic graph (DAG), dictating "
                        "the flow of data through various operations. Since we constrain the DAG of the computation "
                        "graph to have only one input node for each intermediate node, the macro space can be "
                        "described by a list of four integers. The integer of each position represents the input "
                        "source of the operation at the corresponding position in the operation list. For example, "
                        "the integer 0 at position 1 means the corresponding operation at position 1 of the operation "
                        "list uses raw input as input, while the integer 1 at position 3 means the corresponding "
                        "operation at position 3 of the operation list uses the first computing node (the operation "
                        "at position 0 of the operation list) as input. We consider 9 distinct DAG configurations in "
                        "our search space, which are:\n"
                        "- [0, 0, 0, 0]: All operations in the operation list take the raw input directly, creating "
                        "parallel pathways right from the start, allowing for multiple independent transformations of "
                        "the input data.\n"
                        "- [0, 0, 1, 1]: The first two operations in the operation list process the raw input in "
                        "parallel. The third and fourth operations are parallel, both applying transformations to the "
                        "output of the first operation.\n"
                        "- [0, 0, 1, 2]: The first two operations in the operation list are parallel, and the third "
                        "operation processes the output of the first operation. The fourth operation then applies a "
                        "transformation to the output of the second operation, creating a mix of parallel and "
                        "sequential flows.\n"
                        "- [0, 0, 1, 3]: The first two operations in the operation list process the raw input in "
                        "parallel. The third operation processes the output of the first operation. The fourth "
                        "operation extends the sequence by processing the output of the third operation, showcasing a "
                        "blend of parallel processing at the start followed by a sequential chain.\n"
                        "- [0, 1, 1, 1]: The first operation in the operation list processes the raw input, while the "
                        "next three operations process the output of the first operation in parallel, allowing for "
                        "diverse transformations of the same set of features.\n"
                        "- [0, 1, 1, 2]: The first operation in the operation list processes the raw input, while the "
                        "next two operations process the output of the first operation in parallel. The fourth "
                        "operation then processes the output of the second operation, introducing a sequential "
                        "element within a primarily parallel structure.\n"
                        "- [0, 1, 2, 2]: The first operation in the operation list processes the raw input, the "
                        "second operation processes the output of the first operation, and the third and fourth "
                        "operations both apply transformations to the output of the second operation in parallel, "
                        "creating a divergent path after a single sequence.\n"
                        "- [0, 1, 2, 3]: Represents a fully sequential architecture where each operation receives the "
                        "output of the previous operation, forming a linear sequence of transformations from the raw "
                        "input to the final output.\n"
                        "Together, the operation list and the macro architecture list define the computation graph of "
                        "a GNN, including the flow of data through various operations. For example, the model design "
                        "(Architecture: [0, 1, 1, 3], Operations: ['gcn', 'cheb', 'gin', 'fc']) represents a GNN "
                        "architecture where the raw input first undergoes a GCN operation. Subsequently, the output "
                        "of the GCN is processed by the second Chebyshev convolution and the third GIN operations in "
                        "parallel pathways. The fourth operation, the Fully Connected layer, processes the output of "
                        "the GIN operation. The outputs of the second Chebyshev convolution and the Fully Connected "
                        "layer are concatenated together before producing the final output. When seeing a GNN model "
                        "design of this format, you need to understand the actual operations they represent and how "
                        "they are connected."),
             ("user", "{input}")]
        )
        bad_model = any(model.get('bad_models') for model in models_info)
        if bad_model:
            user_input = "Based on the given most similar benchmark dataset and its corresponding top-performing and " \
                         "bad-performing GNN model architectures below, please take a deep breath and work on this " \
                         "problem step-by-step: analyze the potential patterns or underlying principles in the " \
                         "operation lists and the macro architecture lists of the top-performing and bad-performing " \
                         "model designs. This may include commonalities in the choice of operations, preferences for " \
                         "certain macro architecture configurations, or any recurring themes that might indicate a " \
                         "successful or failure approach to constructing GNN architectures for similar types of " \
                         "data. After evaluating these patterns, you need to use your comprehensive knowledge to " \
                         "suggest an optimal model design for the unseen dataset. You should think about how " \
                         "specific operations and macro architecture designs have contributed to high performance " \
                         "in similar datasets. Your suggestion should reflect a thoughtful synthesis of these " \
                         "insights, aiming to capture the most effective elements in the provided top-performing " \
                         "designs and avoid the most ineffective elements in the bad-performing designs. Here are " \
                         "the top-performing and bad-performing designs:\n"
        else:
            user_input = "Based on the given most similar benchmark dataset and its corresponding top-performing GNN " \
                         "model architectures below, please take a deep breath and work on this problem step-by-step: " \
                         "analyze the potential patterns or underlying principles in the operation lists and the macro " \
                         "architecture lists of the top-performing model designs. This may include commonalities in the " \
                         "choice of operations, preferences for certain macro architecture configurations, or any " \
                         "recurring themes that might indicate a successful approach to constructing GNN architectures " \
                         "for similar types of data. After evaluating these patterns, you need to use your comprehensive " \
                         "knowledge to suggest an optimal model design for the unseen dataset. You should think about " \
                         "how specific operations and macro architecture designs have contributed to high performance in " \
                         "similar datasets. Your suggestion should reflect a thoughtful synthesis of these insights, " \
                         "aiming to capture the most effective elements in the provided top-performing designs. Here are " \
                         "the top-performing designs:\n"

        for model_info in models_info:
            selected_dataset = model_info['selected_dataset']
            if similarities and selected_dataset in similarities[dataset_name]:
                user_input += f"Top-performing model designs from {selected_dataset} (Similarity score: {similarities[dataset_name][selected_dataset]}):\n"
            else:
                user_input += f"Top-performing model designs from {selected_dataset}:\n"
            for model_design in model_info['top_models']:
                link_structure, operations = model_design
                user_input += f"- (Architecture: {link_structure}, Operations: {operations})\n"

            if bad_model:
                user_input += f"Bad-performing model designs from {selected_dataset}:\n"
                for model_design in model_info['bad_models']:
                    link_structure, operations = model_design
                    user_input += f"- (Architecture: {link_structure}, Operations: {operations})\n"

        fields = {}
        fields[f"initial_operation"] = (Optional[conlist(str, min_items=4, max_items=4)],
                                                         Field(default=None,
                                                               description=f"The operation list of the optimal model "
                                                                           f"design suggested for the unseen dataset "
                                                                           f"{dataset_name}."))
        fields[f"initial_macro"] = (Optional[conlist(int, min_items=4, max_items=4)],
                                                     Field(default=None,
                                                           description=f"The macro architecture list of the optimal "
                                                                       f"model design suggested for the unseen dataset "
                                                                       f"{dataset_name}."))
        fields[f"initial_design_reason"] = (Optional[str],
                                                             Field(default=None,
                                                                   description=f"Reason for the optimal model design "
                                                                               f"suggested for the unseen dataset "
                                                                               f"{dataset_name}."))
        initialization_tool = create_model('InitialModelDesign', **fields)
        if bad_model:
            initialization_tool.__doc__ = "Suggest an optimal GNN model architecture on the unseen dataset based on the " \
                                          "top-performing and bad-performing GNN model architectures on the most " \
                                          "similar benchmark dataset."
        else:
            initialization_tool.__doc__ = "Suggest an optimal GNN model architecture on the unseen dataset based on the " \
                                          "top-performing GNN model architectures on the most similar benchmark dataset."

        print(prompt)
        print(user_input)

        return prompt, user_input, initialization_tool

    @staticmethod
    def generate_simple_design_suggestion_prompt_parser(dataset_name, description=None):
        """
        Generates a prompt asking the LLM for a model design suggestion.
        :param dataset_name: The source dataset.
        :return: A formatted LLM prompt string.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a machine learning expert proficient in Graph Neural Networks (GNN) design and graph "
                        "dataset understanding. Your task is to recommend a GNN model architecture that performs well "
                        "on the unseen dataset to the user based on the dataset description.\n"
                        "In the context of GNN, the design of a model is described by two main components: the "
                        "operation list and the macro architecture list. Here are the detailed settings:\n"
                        "1. The operation list is a list of four strings. We consider 9 candidate operations, which "
                        "are:\n"
                        "- 'gat': Graph Attention Network layer, utilizing attention mechanisms to weigh the "
                        "importance of nodes' neighbors.\n"
                        "- 'gcn': Graph Convolutional Network layer, applying a convolutional operation over the "
                        "graph to aggregate neighborhood information.\n"
                        "- 'gin': Graph Isomorphism Network layer, designed to capture the graph structure in the "
                        "embedding.\n"
                        "- 'cheb': Chebyshev Spectral Graph Convolution, using Chebyshev polynomials to filter graph "
                        "signals.\n"
                        "- 'sage': GraphSAGE, sampling and aggregating features from a node's neighborhood.\n"
                        "- 'arma': ARMA layer, utilizing Auto-Regressive Moving Average filters for graph "
                        "convolution.\n"
                        "- 'graph': k-GNN, extending the GNN to capture k-order graph motifs.\n"
                        "- 'fc': Fully Connected layer, a dense layer that does not utilize graph structure.\n"
                        "- 'skip': Skip Connection, enabling the creation of residual connections.\n"
                        "For example, an operation list could be ['gcn', 'gin', 'fc', 'cheb'], with 'gcn' as the first "
                        "computing node. The order of operations in the list matters. \n"
                        "2. The macro architecture list is represented as a directed acyclic graph (DAG), dictating "
                        "the flow of data through various operations. Since we constrain the DAG of the computation "
                        "graph to have only one input node for each intermediate node, the macro space can be "
                        "described by a list of four integers. The integer of each position represents the input "
                        "source of the operation at the corresponding position in the operation list. For example, "
                        "the integer 0 at position 1 means the corresponding operation at position 1 of the operation "
                        "list uses raw input as input, while the integer 1 at position 3 means the corresponding "
                        "operation at position 3 of the operation list uses the first computing node (the operation "
                        "at position 0 of the operation list) as input. We consider 9 distinct DAG configurations in "
                        "our search space, which are:\n"
                        "- [0, 0, 0, 0]: All operations in the operation list take the raw input directly, creating "
                        "parallel pathways right from the start, allowing for multiple independent transformations of "
                        "the input data.\n"
                        "- [0, 0, 1, 1]: The first two operations in the operation list process the raw input in "
                        "parallel. The third and fourth operations are parallel, both applying transformations to the "
                        "output of the first operation.\n"
                        "- [0, 0, 1, 2]: The first two operations in the operation list are parallel, and the third "
                        "operation processes the output of the first operation. The fourth operation then applies a "
                        "transformation to the output of the second operation, creating a mix of parallel and "
                        "sequential flows.\n"
                        "- [0, 0, 1, 3]: The first two operations in the operation list process the raw input in "
                        "parallel. The third operation processes the output of the first operation. The fourth "
                        "operation extends the sequence by processing the output of the third operation, showcasing a "
                        "blend of parallel processing at the start followed by a sequential chain.\n"
                        "- [0, 1, 1, 1]: The first operation in the operation list processes the raw input, while the "
                        "next three operations process the output of the first operation in parallel, allowing for "
                        "diverse transformations of the same set of features.\n"
                        "- [0, 1, 1, 2]: The first operation in the operation list processes the raw input, while the "
                        "next two operations process the output of the first operation in parallel. The fourth "
                        "operation then processes the output of the second operation, introducing a sequential "
                        "element within a primarily parallel structure.\n"
                        "- [0, 1, 2, 2]: The first operation in the operation list processes the raw input, the "
                        "second operation processes the output of the first operation, and the third and fourth "
                        "operations both apply transformations to the output of the second operation in parallel, "
                        "creating a divergent path after a single sequence.\n"
                        "- [0, 1, 2, 3]: Represents a fully sequential architecture where each operation receives the "
                        "output of the previous operation, forming a linear sequence of transformations from the raw "
                        "input to the final output.\n"
                        "Together, the operation list and the macro architecture list define the computation graph of "
                        "a GNN, including the flow of data through various operations. For example, the model design "
                        "(Architecture: [0, 1, 1, 3], Operations: ['gcn', 'cheb', 'gin', 'fc']) represents a GNN "
                        "architecture where the raw input first undergoes a GCN operation. Subsequently, the output "
                        "of the GCN is processed by the second Chebyshev convolution and the third GIN operations in "
                        "parallel pathways. The fourth operation, the Fully Connected layer, processes the output of "
                        "the GIN operation. The outputs of the second Chebyshev convolution and the Fully Connected "
                        "layer are concatenated together before producing the final output. When seeing a GNN model "
                        "design of this format, you need to understand the actual operations they represent and how "
                        "they are connected."),
             ("user", "{input}")]
        )
        user_input = "Based on the following dataset description, please take a deep breath and work on this problem " \
                     "step-by-step: use your comprehensive knowledge to suggest an optimal model design for the " \
                     "unseen dataset. You should think about how specific operations and macro architecture designs " \
                     "could potentially contribute to high performance on the unseen dataset. Here is the dataset " \
                     "description:\n"
        user_input += description

        fields = {}
        fields[f"initial_operation"] = (Optional[conlist(str, min_items=4, max_items=4)],
                                                         Field(default=None,
                                                               description=f"The operation list of the optimal model "
                                                                           f"design suggested for the unseen dataset."))
        fields[f"initial_macro"] = (Optional[conlist(int, min_items=4, max_items=4)],
                                                     Field(default=None,
                                                           description=f"The macro architecture list of the optimal "
                                                                       f"model design suggested for the unseen dataset."))
        fields[f"initial_design_reason"] = (Optional[str],
                                                             Field(default=None,
                                                                   description=f"Reason for the optimal model design "
                                                                               f"suggested for the unseen dataset."))
        initialization_tool = create_model('InitialModelDesign', **fields)
        initialization_tool.__doc__ = "Suggest an optimal GNN model architecture on the unseen dataset based on the " \
                                      "dataset description."

        return prompt, user_input, initialization_tool

    @staticmethod
    def generate_design_suggestion_prompt(dataset_name, models_info, similarities=None, description=None):
        """
        Generates a prompt asking the LLM for a model design suggestion.
        :param dataset_name: The source dataset.
        :param models_info: The top model designs.
        :param similarities: Optional dictionary of similarity scores.
        :return: A formatted LLM prompt string.
        """
        # Introduction to model design components
        if models_info:
            bad_model = any(model.get('bad_models') for model in models_info)
            if bad_model:
                intro = ("The task at hand involves leveraging the best model design knowledge and practices from similar benchmark datasets in the field of Graph Neural Networks (GNN). By examining top-performing and bad-performing models on these datasets, we aim to quickly recommend an optimal model design for an unseen dataset, ensuring good performance with minimal initial experimentation.")
            else:
                intro = ("The task at hand involves leveraging the best model design knowledge and practices from similar benchmark datasets in the field of Graph Neural Networks (GNN). By examining top-performing models on these datasets, we aim to quickly recommend an optimal model design for an unseen dataset, ensuring good performance with minimal initial experimentation.")
        else:
            intro = ("The task at hand involves leveraging the best model design knowledge and practices from your knowledge in the field of Graph Neural Networks (GNN). By examining the textual description of the unseen dataset, we aim to quickly recommend an optimal model design for an unseen dataset, ensuring good performance with minimal initial experimentation.")
        
        intro += ("\nIn the context of GNN, the design of a model is described by two main components:\n"
                  "1. The macro architecture and the operations applied at each node. The macro architecture "
                  "is represented as a directed acyclic graph (DAG), dictating the flow of data through various "
                  "operations. Since we constrain the DAG of the computation graph to have only one input node for each"
                  " intermediate node, the macro space can be described by a list of integers, indicating the input "
                  "node index for each computing node (0 for the raw input, 1 for the first computing node, etc.) We "
                  "consider 9 distinct DAG configurations in our search space:\n"
                  "- [0, 0, 0, 0]: All operations take the raw input directly, creating parallel pathways right from "
                  "the start, allowing for multiple independent transformations of the input data.\n"
                  "- [0, 0, 0, 1]: The first three operations are parallel, directly taking the raw input. The fourth "
                  "operation processes the output of the first operation, introducing a sequential step after parallel "
                  "processing.\n"
                  "- [0, 0, 1, 1]: The first two operations process the raw input in parallel. The third and fourth "
                  "operations are parallel, both applying transformations to the output of the first operation.\n"
                  "- [0, 0, 1, 2]: The first two operations are parallel, and the third operation processes the output "
                  "of the first. The fourth operation then applies a transformation to the output of the second, "
                  "creating a mix of parallel and sequential flows.\n"
                  "- [0, 0, 1, 3]: This starts with two operations processing the raw input in parallel. The third "
                  "operation processes the output of the first operation. The fourth operation extends the sequence by "
                  "processing the output of the third operation, showcasing a blend of parallel processing at the start"
                  " followed by a sequential chain.\n"
                  "- [0, 1, 1, 1]: The first operation processes the raw input, while the next three operations "
                  "process the output of the first operation in parallel, allowing for diverse transformations of the "
                  "same set of features.\n"
                  "- [0, 1, 1, 2]: After the raw input is processed by the first operation, the next two operations "
                  "work in parallel on this output. The fourth operation then processes the output of the second "
                  "operation, introducing a sequential element within a primarily parallel structure.\n"
                  "- [0, 1, 2, 2]: The first operation processes the raw input, the second operation processes its "
                  "output, and the third and fourth operations both apply transformations to the output of the second "
                  "operation in parallel, creating a divergent path after a single sequence.\n"
                  "- [0, 1, 2, 3]: Represents a fully sequential architecture where each operation receives the output "
                  "of the previous operation, forming a linear sequence of transformations from the raw input to the "
                  "final output. This structure allows for a complex, layered processing of features.\n"
                  "These architectures allow for varied feature transformations and combinations, reflecting the "
                  "complexity and adaptability required in GNN models to effectively process graph-structured data.\n"
                  "2. The operations applied at each node, specified by a list of strings. We consider 9 candidate "
                  "operations, which are:\n"
                  "- 'gat': Graph Attention Network layer, utilizing attention mechanisms to weigh the importance "
                  "of nodes' neighbors.\n"
                  "- 'gcn': Graph Convolutional Network layer, applying a convolutional operation over the graph "
                  "to aggregate neighborhood information.\n"
                  "- 'gin': Graph Isomorphism Network layer, designed to capture the graph structure in the embedding.\n"
                  "- 'cheb': Chebyshev Spectral Graph Convolution, using Chebyshev polynomials to filter graph signals.\n"
                  "- 'sage': GraphSAGE, sampling and aggregating features from a node's neighborhood.\n"
                  "- 'arma': ARMA layer, utilizing Auto-Regressive Moving Average filters for graph convolution.\n"
                  "- 'graph': k-GNN, extending the GNN to capture k-order graph motifs.\n"
                  "- 'fc': Fully Connected layer, a dense layer that does not utilize graph structure.\n"
                  "- 'skip': Skip Connection, enabling the creation of residual connections.\n"
                  "Together, these components define the computation graph of a GNN, including the flow of "
                  "data through various operations. For example, the model design [0, 1, 1, 3] with operations "
                  "['gcn', 'cheb', 'gin', 'fc'] implies an architecture where the input first undergoes a GCN operation"
                  ". Subsequently, the output of the GCN is processed by Chebyshev convolution and GIN operations in "
                  "parallel pathways. The final operation, Fully Connected layer, processes the output of the GIN "
                  "operation. The outputs of the FC and Chebyshev convolutions are concatenated before producing the "
                  "final output.\n")

        if models_info:
            if bad_model:
                prompt = "You will need to recommend an optimal model design for the unseen dataset based on the following top and bad model designs from similar datasets. Here are the top and bad model designs gathered from similar benchmark datasets:\n"
            else:
                prompt = "You will need to recommend an optimal model design for the unseen dataset based on the following top model designs from similar datasets. Here are the top model designs gathered from similar benchmark datasets:\n"

            prompt += f"For the unseen dataset:\n"
            '''
            for model_info in models_info:
                selected_dataset = model_info['selected_dataset']
                if similarities and selected_dataset in similarities[dataset_name]:
                    prompt += f"Similarity score to {selected_dataset}: {similarities[dataset_name][selected_dataset]}\n"
                for model_design in model_info['top_models']:
                    link_structure, operations = model_design
                    prompt += f"- From '{selected_dataset}': (Architecture: {link_structure}, Operations: {operations})\n"

                if len(model_info['bad_models']) > 0:
                    prompt += f"Here are the bad model designs from {selected_dataset} that may not perform well:\n"
                    for model_design in model_info['bad_models']:
                        link_structure, operations = model_design
                        prompt += f"- From '{selected_dataset}': (Architecture: {link_structure}, Operations: {operations})\n"
                prompt += "\n"
            '''
            for model_info in models_info:
                selected_dataset = model_info['selected_dataset']
                if similarities and selected_dataset in similarities[dataset_name]:
                    prompt += f"Top-performing model designs from {selected_dataset} (Similarity score: {similarities[dataset_name][selected_dataset]}):\n"
                else:
                    prompt += f"Top-performing model designs from {selected_dataset}:\n"
                
                # List out top-performing model designs
                for model_design in model_info['top_models']:
                    link_structure, operations = model_design
                    prompt += f"- (Architecture: {link_structure}, Operations: {operations})\n"

                # List out bad-performing model designs
                if bad_model:
                    prompt += f"Bad-performing model designs from {selected_dataset}:\n"
                    for model_design in model_info['bad_models']:
                        link_structure, operations = model_design
                        prompt += f"- (Architecture: {link_structure}, Operations: {operations})\n"
        else:
            prompt = "You will need to recommend an optimal model design for the unseen dataset based on the following description: "
            if description:
                prompt += description + '\n'

        if models_info:
            if bad_model:
                prompt += ("Based on the insights from similar benchmark datasets, consider the potential patterns or underlying principles in the top and bad model designs. This includes commonalities in the choice of operations, preferences for certain macro architecture configurations, or any recurring themes that might indicate a successful approach to constructing GNN architectures for similar types of data. Evaluate these patterns and, using your comprehensive analysis, suggest an optimal model design for the source dataset. Consider how specific operations and architecture designs have contributed to high performance in similar datasets. Your suggestion should reflect a thoughtful synthesis of these insights, aiming to capture the most effective elements of the provided designs and avoid the most ineffective elements. Additionally, pay attention to the similarity scores between datasets, if provided, to gauge the relevance of each design's features in relation to the source dataset.\n")
            else:
                prompt += ("Based on the insights from similar benchmark datasets, consider the potential patterns or underlying principles in the top model designs. This includes commonalities in the choice of operations, preferences for certain macro architecture configurations, or any recurring themes that might indicate a successful approach to constructing GNN architectures for similar types of data. Evaluate these patterns and, using your comprehensive analysis, suggest an optimal model design for the source dataset. Consider how specific operations and architecture designs have contributed to high performance in similar datasets. Your suggestion should reflect a thoughtful synthesis of these insights, aiming to capture the most effective elements of the provided designs. Additionally, pay attention to the similarity scores between datasets, if provided, to gauge the relevance of each design's features in relation to the source dataset.\n")
            
            prompt += ("Now, please provide a suggested architecture and set of operations for the source dataset, "
                       "tailoring each recommendation to maximize potential performance based on the observed design "
                       "patterns.")
        else:
            prompt += ("Now, please provide a suggested architecture and set of operations for the source dataset, "
                       "tailoring each recommendation to maximize potential performance based on your knowledge.")

        prompt += ("Your suggested optimal model design for the source dataset should be in the same search space we "
                   "defined. Your answer should be in the following format:\n")
        prompt += f"For the unseen dataset: (Architecture: [TBD], Operations: [TBD])\nReasons:\n"

        return intro + prompt
    
    def generate_llm_mutation_prompt(self, dataset_name, promoted_child, current_design, generation, gnas_history, 
                                     best_design, detailed_log, candidate_pools):
        """
        Generate a prompt for the LLM to help refine the promoted child model based on the provided GNAS context.

        :param dataset_name: Name of the dataset being optimized.
        :param promoted_child: The child model selected for further mutation.
        :param current_design: Current model design before the generation of children.
        :param generation: Current generation number in the evolutionary search.
        :param gnas_history: Historical record of all generations and their model performances.
        :param best_design: The best model design encountered so far in terms of performance.
        :param detailed_log: Flag to include detailed training logs in the prompt.
        :param candidate_pools: Information about top-performing designs from the most similar dataset.
        :return: A string prompt for the LLM.
        """
        intro = self.generate_GNAS_task_description() + self.generate_short_space_description()

        # Building the history narrative
        history = f"Currently, you are the evolutionary Graph NAS agent at {generation} generation. We have already explored various Graph Neural Network architectures to optimize performance. Your further recommendation should not repeat any of the models in the optimization trajectory (history) below:\n"

        # Iterate over the history dictionary, sorted by iteration keys to maintain order
        for iter_num in sorted(gnas_history.keys(), key=int):
            if iter_num == '0':
                detail_list = gnas_history[iter_num]
                history += f"Generation {iter_num} tested {len(detail_list)} children:\n"
                for details in detail_list:
                    history += f" - Achieved a performance of {round(details['perf'], 3)} with operations {details['ops']} and macro architecture {details['link']}.\n"
            else:
                details = gnas_history[iter_num]
                history += f" - Generation {iter_num} achieved a performance of {round(details['perf'])} with operations {details['ops']} and macro architecture {details['link']}.\n"

        # Highlighting the best model so far
        history += f"The best model design so far is operations {best_design['ops']} and macro architecture {best_design['link']}, which achieved a performance of {round(best_design['perf'])} at generation {best_design['iteration']}.\n"


        # If detailed logs are available, add them to the prompt
        log = ""
        if detailed_log:
            log = f"Here is the training log snapshot (every 25 epochs) of the last experiment with operations {current_design['ops']} and macro architecture {current_design['link']}:\n"
            for log_entry in current_design["detailed_log"]:
                log += f"Epoch {log_entry['epoch']}: Train Acc - {log_entry['train_accuracy']}, Val Acc - {log_entry['val_accuracy']}, Test Acc - {log_entry['test_accuracy']}, Train Loss - {log_entry['train_loss']}, Val Loss - {log_entry['val_loss']}, Test Loss - {log_entry['test_loss']};\n"

        # Include insights from candidate pools if available
        if candidate_pools:
            top_models = candidate_pools[dataset_name][0]
            selected_dataset = top_models['selected_dataset']
            top_models = top_models['top_models']
            knowledge = f"\nAdditionally, insights into top-performing designs in the most similar benchmark dataset {selected_dataset} include:\n"
            for model in top_models:
                knowledge += f" - Architecture: {model[0]}, Operations: {model[1]}\n"
        else:
            knowledge = ""

        # Children details for selection
        #prompt += "We have completed the crossover on the best child from the last generation with respect to the top " \
        #         "model designs from the second and third similar datasets. Here is the current generation of " \
        #         "children for selection:\n"
        instruction = f"\nWe have completed the exploration (crossover) on the best design so far with respect to the top model designs from the second and third similar datasets. Here is the promoted child that has the highest empirical performance on the most similar dataset:\n"
        instruction += f"Promoted Child: Architecture {promoted_child['link']}, Operations {promoted_child['ops']}\n"

        # Finally, ask for suggestions on improvements
        instruction += f"\nAs an optimal Graph NAS that performs exploitation (mutation), please further refine (mutate) this promoted child ({promoted_child['link']}, {promoted_child['ops']}) based on its potential effectiveness, the experiment history"
        #prompt += "\nAs an optimal Graph NAS, please suggest the best child from the current generation for further " \
        #          "validation based on their potential effectiveness, the history of experimental performances"
        if detailed_log and candidate_pools:
            instruction += ", training log of last trial, and the potential pattern of top-performing designs in the most similar datasets."
        elif detailed_log:
            instruction += " and training log of last trial."
        elif candidate_pools:
            instruction += " and the potential pattern of top-performing designs in the most similar datasets."
        else:
            instruction += "."
        instruction += "The objective is to maximize the model's performance. You should modify upon the promoted child and shouldn't repropose model designs that have already been validated in the optimization trajectory. Your suggested optimal model design for the unseen dataset should be in the same search space we defined and should not repeat any model design already contained in the experiment history. Your answer should closely follow the output format below:\n\n"
        instruction += f"Response Format:\nFor the unseen dataset, I recommend (Architecture: [TBD], Operations: [TBD]).\nReasons for recommendation: TBD\n"

        return intro + history + log + knowledge + instruction
    
    @staticmethod
    def generate_GNAS_task_description():
        return "You are an expert in the field of neural architecture search. Your task is to perform the neural architecture search of the Graph Neural Network on the unseen graph dataset. "

    @staticmethod
    def generate_short_space_description():
        return "To recall, in the context of GNN, the design of a model is described by two main components:\n1. The macro architecture list defines how the operations are connected in a directed acyclic graph (DAG). It is specified as a list of four integers where each integer denotes the input source (0 for the raw input, 1 for the first computing node, etc) for the corresponding operation in the operation list. This structure allows the defining of various computational graph architectures, which can be sequential, parallel, or mixed. We consider 9 distinct DAG configurations in our search space: [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3], [0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 2], [0, 1, 2, 3].\n2. The operation list consists of a set of operations that can be used in constructing a Graph Neural Network (GNN). We consider 9 candidate operations of GNN, which are: 'gat', 'gcn', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip'. \nTogether, these components define the computation graph of a GNN, including the flow of data through various operations. You need to understand the real structure of the GNNs given its macro architecture list and operation list.\n"

    @staticmethod
    def extract_model_designs(llm_response, dataset_name):
        """
        Extracts model designs suggested by the LLM for each source dataset.

        :param llm_response: A string containing the LLM's response in the specified format.
        :return: A dictionary with source dataset names as keys and their suggested model designs as values.
        """
        # Pattern to match the format of the LLM's response for each dataset
        pattern = r"\(Architecture: (\[.*?\]), Operations: (\[.*?\])\)"

        # Find all matches in the response
        matches = re.findall(pattern, llm_response)

        # Initialize a dictionary to hold the extracted designs
        suggested_designs = {}

        # Iterate through all matches and populate the dictionary
        for match in matches:
            architecture, operations = match
            suggested_designs[dataset_name] = {
                "link": eval(architecture),  # Use eval to convert string representation of list to actual list
                "ops": eval(operations)
            }

        return suggested_designs

    @staticmethod
    def has_bad_models(models_info):
        for model in models_info:
            if 'bad_models' in model and model['bad_models']:
                return True
        return False

