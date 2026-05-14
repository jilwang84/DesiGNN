# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

import ast
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain_core.pydantic_v1 import Field, create_model, conlist
except ModuleNotFoundError:
    from pydantic import Field, create_model
    from pydantic import conlist as _pydantic_conlist

    def conlist(item_type, min_items=None, max_items=None, **kwargs):
        if min_items is not None:
            kwargs["min_length"] = min_items
        if max_items is not None:
            kwargs["max_length"] = max_items
        return _pydantic_conlist(item_type, **kwargs)


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
    def generate_design_suggestion_prompt_parser_new(dataset_name, models_info, similarities=None):
        """
        Generates a prompt asking the LLM for a model design suggestion.
        :param dataset_name: The source dataset.
        :param models_info: The top model designs.
        :param similarities: Optional dictionary of similarity scores.
        :return: A formatted LLM prompt string.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", "The task at hand involves leveraging the best model design knowledge and practices from similar "
                        "benchmark datasets in the field of Graph Neural Networks (GNN). By examining top-performing "
                        "models on these datasets, we aim to quickly recommend an optimal model design for an unseen "
                        "dataset, ensuring good performance with minimal initial experimentation. "
                        "\nIn the context of GNN, the design of a model is described by two main components:\n"
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
                        "final output. This structure allows for a complex, layered processing of features."
                        "\nThese architectures allow for varied feature transformations and combinations, reflecting the "
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
                        "final output.\n\n"),
             ("user", "{input}")]
        )
        user_input = "You will need to recommend an optimal model design for the unseen dataset based on the " \
                     "following top model designs from similar datasets. Here are the top model designs gathered " \
                     "from similar benchmark datasets:\n"

        user_input += f"For the unseen dataset:\n"
        for model_info in models_info:
            selected_dataset = model_info['selected_dataset']
            if similarities and selected_dataset in similarities[dataset_name]:
                user_input += f"Similarity score to {selected_dataset}: {similarities[dataset_name][selected_dataset]}\n"
            for model_design in model_info['top_models']:
                link_structure, operations = model_design
                user_input += f"- From '{selected_dataset}': (Architecture: {link_structure}, Operations: {operations})\n"

        if models_info:
            user_input += ("Based on the insights from similar benchmark datasets, consider the potential patterns "
                       "or underlying principles in the top model designs. This includes commonalities in the choice "
                       "of operations, preferences for certain macro architecture configurations, or any recurring "
                       "themes that might indicate a successful approach to constructing GNN architectures for similar"
                       " types of data. Evaluate these patterns and, using your comprehensive analysis, suggest an "
                       "optimal model design for the source dataset. Consider how specific operations and architecture"
                       " designs have contributed to high performance in similar datasets. Your suggestion should "
                       "reflect a thoughtful synthesis of these insights, aiming to capture the most effective "
                       "elements of the provided designs. Additionally, pay attention to the similarity scores between"
                       " datasets, if provided, to gauge the relevance of each design's features in relation to the "
                       "source dataset.\n")

        if models_info:
            user_input += ("Now, please provide a suggested architecture and set of operations for the source dataset, "
                       "tailoring each recommendation to maximize potential performance based on the observed design "
                       "patterns.")
        else:
            user_input += ("Now, please provide a suggested architecture and set of operations for the source dataset, "
                       "tailoring each recommendation to maximize potential performance based on your knowledge.")

        user_input += ("Your suggested optimal model design for the source dataset should be in the same search space we "
                   "defined. Your answer should be in the following format:\n")
        user_input += f"For the unseen dataset: (Architecture: [TBD], Operations: [TBD])\nReasons:\n"

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
        initialization_tool.__doc__ = "Suggest an optimal GNN model architecture on the unseen dataset based on the " \
                                      "top-performing GNN model architectures on the most similar benchmark dataset."

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

    @staticmethod
    def generate_design_suggestion_prompt_new(dataset_name, models_info, similarities=None):
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
                        "on the unseen dataset based on the top-performing GNN model architectures on the most similar "
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
            for model_design in model_info['top_models']:
                link_structure, operations = model_design
                user_input += f"- (Architecture: {link_structure}, Operations: {operations})\n"

        for model_info in models_info:
            selected_dataset = model_info['selected_dataset']
            if similarities and selected_dataset in similarities[dataset_name]:
                user_input += f"Bad-performing model designs from {selected_dataset} (Similarity score: {similarities[dataset_name][selected_dataset]}):\n"
            for model_design in model_info['bad_models']:
                link_structure, operations = model_design
                user_input += f"- (Architecture: {link_structure}, Operations: {operations})\n"

        user_input += "Your suggested optimal model design for the unseen dataset should be in the same search space " \
                      "we defined. Your answer should be in the following format:\nFor the unseen dataset: " \
                      "(Architecture: [TBD], Operations: [TBD])\nReasons:\n"

        return prompt, user_input

    @staticmethod
    def generate_design_refinement_prompt_parser(dataset_name, current_design, iteration, gnas_history, best_design,
                                                 detailed_log=False, candidate_pools=None):
        """
        Generate a new LLM prompt to suggest design improvements based on performance and optionally detailed training logs.

        :param dataset_name: Unseen dataset name.
        :param current_design: Dictionary of the current design.
        :param iteration: The current iteration.
        :param gnas_history: The Graph NAS history.
        :param best_design: Dictionary of the best design.
        :param detailed_log: Use detailed training log as context or not.
        :param candidate_pools: Information about top-performing designs from similar datasets.
        :return: A string prompt for the LLM.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a machine learning expert proficient in Graph Neural Networks (GNN) design and graph "
                        "dataset understanding. Your task is to perform a neural architecture search of GNN on the "
                        f"unseen graph dataset {dataset_name} based on the optimization trajectory and top-performing "
                        "GNN model architectures on the most similar benchmark dataset.\n"
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
        user_input = f"Currently, you are the Graph NAS agent at {iteration} iteration. We have explored various GNN " \
                     f"architectures to optimize performance on the unseen dataset. Here's the history:\n"

        # Iterate over the history dictionary, sorted by iteration keys to maintain order
        for iter_num in sorted(gnas_history.keys(), key=int):
            # Access the details of each iteration
            details = gnas_history[iter_num]
            user_input += f" - Iteration {iter_num} achieved a performance of {details['perf']} with operations " \
                          f"{details['ops']} and macro architecture {details['link']}.\n"

        # Highlighting the best model so far
        user_input += f"The best model design so far is operations {best_design['ops']} and macro architecture " \
                      f"{best_design['link']}, which achieved a performance of {best_design['perf']} at iteration " \
                      f"{best_design['iteration']}.\n"

        # Adding the performance of the most recent model
        user_input += f"The most recent model design, which tested operations {current_design['ops']} and macro " \
                      f"architecture {current_design['link']}, achieved a performance of {current_design['perf']}.\n"

        # If detailed logs are available, add them to the prompt
        if detailed_log:
            user_input += "Here is a summary of the most recent training log over every 25 epochs:\n"
            for log_entry in current_design["detailed_log"]:
                user_input += (f"Epoch {log_entry['epoch']}: Train Acc: {log_entry['train_accuracy']}, "
                               f"Val Acc: {log_entry['val_accuracy']}, Test Acc: {log_entry['test_accuracy']}, "
                               f"Train Loss: {log_entry['train_loss']}, Val Loss: {log_entry['val_loss']}, "
                               f"Test Loss: {log_entry['test_loss']}, Latency: {log_entry['latency']}s;\n")

        # Include insights from candidate pools if available
        if candidate_pools:
            user_input += "\nAdditionally, please consider insights from top-performing designs in similar benchmark " \
                          "datasets: \n"
            for pool_key, pool in candidate_pools.items():
                for dataset_info in pool:
                    selected_dataset = dataset_info['selected_dataset']
                    top_models = dataset_info['top_models']
                    user_input += f"From similar dataset '{selected_dataset}', top model designs include:\n"
                    for model in top_models:
                        architecture, operations = model
                        user_input += f"  - Architecture: {architecture}, Operations: {operations}\n"

        # Finally, ask for suggestions on improvements
        user_input += "\nYour objective is to maximize the model's performance on the unseen dataset " \
                      f"{dataset_name}. As an optimal Graph NAS agent, please suggest modifications to the model " \
                      "design (the operation list and the macro architecture list) to enhance the model's " \
                      "performance for the next trial based on the history of experimental performances"
        if detailed_log and candidate_pools:
            user_input += ", training log of last trial, and top-performing designs in similar benchmark datasets."
        elif detailed_log:
            user_input += " and training log of last trial."
        elif candidate_pools:
            user_input += " and top-performing designs in similar benchmark datasets."
        else:
            user_input += "."
        user_input += "You shouldn’t repropose model designs that have already been validated in the optimization " \
                      "trajectory. "

        fields = {}
        fields[f"{dataset_name}_refined_operation"] = (Optional[conlist(str, min_items=4, max_items=4)],
                                                       Field(default=None,
                                                             description=f"The operation list of the refined model "
                                                                         f"design suggested for the unseen dataset "
                                                                         f"{dataset_name} as the next trail."))
        fields[f"{dataset_name}_refined_macro"] = (Optional[conlist(int, min_items=4, max_items=4)],
                                                   Field(default=None,
                                                         description=f"The macro architecture list of the refined "
                                                                     f"model design suggested for the unseen dataset "
                                                                     f"{dataset_name} as the next trail."))
        fields[f"{dataset_name}_refined_design_reason"] = (Optional[str],
                                                           Field(default=None,
                                                                 description=f"Reason for the refined model design "
                                                                             f"suggested for the unseen dataset "
                                                                             f"{dataset_name} as the next trail."))
        optimization_tool = create_model('RefinedModelDesign', **fields)
        optimization_tool.__doc__ = "Suggest a better GNN model architecture on the unseen dataset based on the " \
                                    "optimization trajectory and top-performing GNN model architectures on the most " \
                                    "similar benchmark dataset."

        return prompt, user_input, optimization_tool

    @staticmethod
    def generate_design_refinement_prompt(dataset_name, current_design, iteration, gnas_history, best_design,
                                          detailed_log=False, candidate_pools=None):
        """
        Generate a new LLM prompt to suggest design improvements based on performance and optionally detailed training logs.

        :param dataset_name: Unseen dataset name.
        :param current_design: Dictionary of the current design.
        :param iteration: The current iteration.
        :param gnas_history: The Graph NAS history.
        :param best_design: Dictionary of the best design.
        :param detailed_log: Use detailed training log as context or not.
        :param candidate_pools: Information about top-performing designs from similar datasets.
        :return: A string prompt for the LLM.
        """
        intro = (f"You are an expert in the field of neural architecture search. Your task is to perform neural "
                 f"architecture search of Graph Neural Network on the unseen graph dataset {dataset_name}. To recall, "
                 f"in the context of GNN, the design of a model is described by two main components:\n"
                 f"1. The macro architecture is represented as a directed acyclic graph (DAG), dictating the flow of "
                 f"data through various operations. The macro space can be described by a list of integers, indicating "
                 f"the input node index for each computing node (0 for the raw input, 1 for the first computing node, "
                 f"etc.) We consider 9 distinct DAG configurations in our search space: [0, 0, 0, 0], [0, 0, 0, 1], "
                 f"[0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3], [0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 2], [0, 1, 2, 3].\n"
                 f"2. The operations applied at each node, specified by a list of strings. We consider 9 candidate "
                 f"operations, which are: 'gat', 'gcn', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip'. \n"
                 f"Together, these components define the computation graph of a GNN, including the flow of data through"
                 f" various operations. The meaning behind each component of this search space has been introduced "
                 f"before.\n")

        # Building the history narrative
        prompt = f"Currently, you are the Graph NAS agent at {iteration} iteration. We have explored various Graph " \
                 f"Neural Network architectures to optimize performance. Here's the history:\n"
        # Iterate over the history dictionary, sorted by iteration keys to maintain order
        for iter_num in sorted(gnas_history.keys(), key=int):
            # Access the details of each iteration
            details = gnas_history[iter_num]
            prompt += f" - Iteration {iter_num} achieved a performance of {details['perf']} with operations " \
                      f"{details['ops']} and macro architecture {details['link']}.\n"

        # Highlighting the best model so far
        prompt += f"The best model design so far is operations {best_design['ops']} and macro architecture " \
                  f"{best_design['link']}, which achieved a performance of {best_design['perf']} at iteration " \
                  f"{best_design['iteration']}.\n"

        # Adding the performance of the most recent model
        prompt += f"The most recent model design, which tested operations {current_design['ops']} and macro " \
                  f"architecture {current_design['link']}, achieved a performance of {current_design['perf']}.\n"

        # If detailed logs are available, add them to the prompt
        if detailed_log:
            prompt += "Here is a summary of the training log over every 25 epochs:\n"
            for log_entry in current_design["detailed_log"]:
                prompt += (f"Epoch {log_entry['epoch']}: Train Acc: {log_entry['train_accuracy']}, "
                           f"Val Acc: {log_entry['val_accuracy']}, Test Acc: {log_entry['test_accuracy']}, "
                           f"Train Loss: {log_entry['train_loss']}, Val Loss: {log_entry['val_loss']}, "
                           f"Test Loss: {log_entry['test_loss']}, Latency: {log_entry['latency']}s;\n")

        # Include insights from candidate pools if available
        if candidate_pools:
            prompt += "\nAdditionally, insights from top-performing designs in similar datasets include:\n"
            for pool_key, pool in candidate_pools.items():
                for dataset_info in pool:
                    selected_dataset = dataset_info['selected_dataset']
                    top_models = dataset_info['top_models']
                    prompt += f"From similar dataset '{selected_dataset}', top model designs include:\n"
                    for model in top_models:
                        architecture, operations = model
                        prompt += f"  - Architecture: {architecture}, Operations: {operations}\n"

        # Finally, ask for suggestions on improvements
        prompt += "\nAs an optimal Graph NAS, please suggest improvements or modifications to enhance the model's " \
                  "performance for the next trial based on the history of experimental performances"
        if detailed_log and candidate_pools:
            prompt += ", training log of last trial, and top-performing designs in similar datasets."
        elif detailed_log:
            prompt += " and training log of last trial."
        elif candidate_pools:
            prompt += " and top-performing designs in similar datasets."
        else:
            prompt += "."
        prompt += f"The objective is to maximize the model's performance. You shouldn’t repropose model designs that " \
                  f"have already been validated in the optimization trajectory.Your suggested optimal model design " \
                  f"for the unseen dataset {dataset_name} should be in the same search space we defined. Your answer " \
                  f"should be in the following format:\n"
        prompt += f"For ‘{dataset_name}’: (Architecture: [TBD], Operations: [TBD])\nReasons:\n"

        return intro + prompt

    @staticmethod
    def generate_design_evolution_prompt_parser(dataset_name, num_children, current_design, generation, gnas_history,
                                                best_design, detailed_log=False, candidate_pools=None):
        """
        Generate a new LLM prompt to suggest design improvements based on performance and optionally detailed training logs.

        :param dataset_name: Unseen dataset name.
        :param current_design: Dictionary of the current design.
        :param iteration: The current iteration.
        :param gnas_history: The Graph NAS history.
        :param best_design: Dictionary of the best design.
        :param detailed_log: Use detailed training log as context or not.
        :param candidate_pools: Information about top-performing designs from similar datasets.
        :return: A string prompt for the LLM.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a machine learning expert proficient in Graph Neural Networks (GNN) design and graph "
                        "dataset understanding. Your task is to perform an evolutionary neural architecture search of "
                        f"GNN on the unseen graph dataset {dataset_name} based on the optimization trajectory and "
                        "top-performing GNN model architectures on the most similar benchmark dataset.\n"
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
        user_input = f"Currently, you are the evolutionary Graph NAS agent at {generation} generation. We have " \
                     f"explored various GNN architectures to optimize performance on the unseen dataset. Here's the " \
                     f"history:\n"

        # Iterate over the history dictionary, sorted by iteration keys to maintain order
        for iter_num in sorted(gnas_history.keys(), key=int):
            # Access the details of each iteration
            details = gnas_history[iter_num]
            user_input += f"Generation {iter_num} tested {len(details)} children:\n"
            for child in details:
                user_input += f" - Operations {child['ops']} and macro architecture {child['link']} achieved a " \
                              f"performance of {child['perf']}.\n"

        # Highlighting the best model so far
        user_input += f"The best model design so far is operations {best_design['ops']} and macro architecture " \
                      f"{best_design['link']}, which achieved a performance of {best_design['perf']} at iteration " \
                      f"{best_design['iteration']}.\n"

        # Adding the performance of the most recent model
        user_input += f"The best child in last generation, which tested operations {current_design['ops']} and macro " \
                      f"architecture {current_design['link']}, achieved a performance of {current_design['perf']}.\n"

        # If detailed logs are available, add them to the prompt
        if detailed_log:
            user_input += "Here is a summary of its training log over every 25 epochs:\n"
            for log_entry in current_design["detailed_log"]:
                user_input += (f"Epoch {log_entry['epoch']}: Train Acc: {log_entry['train_accuracy']}, "
                               f"Val Acc: {log_entry['val_accuracy']}, Test Acc: {log_entry['test_accuracy']}, "
                               f"Train Loss: {log_entry['train_loss']}, Val Loss: {log_entry['val_loss']}, "
                               f"Test Loss: {log_entry['test_loss']}, Latency: {log_entry['latency']}s;\n")

        # Include insights from candidate pools if available
        if candidate_pools:
            user_input += "\nAdditionally, please consider insights from top-performing designs in similar benchmark " \
                          "datasets: \n"
            for pool_key, pool in candidate_pools.items():
                for dataset_info in pool:
                    selected_dataset = dataset_info['selected_dataset']
                    top_models = dataset_info['top_models']
                    user_input += f"From similar dataset '{selected_dataset}', top model designs include:\n"
                    for model in top_models:
                        architecture, operations = model
                        user_input += f"  - Architecture: {architecture}, Operations: {operations}\n"

        # Finally, ask for suggestions on improvements
        user_input += f"\nYour objective is to maximize the model's performance on the unseen dataset. As an optimal " \
                      f"evolutionary Graph NAS agent, please suggest modifications to the model design (the " \
                      f"operation list and the macro architecture list) to enhance the model's performance for the " \
                      f"next generation ({num_children} children models in total) based on the best child from the " \
                      f"last generation, the history of experimental performances"
        if detailed_log and candidate_pools:
            user_input += ", training log of last best child, and top-performing designs in similar datasets."
        elif detailed_log:
            user_input += " and training log of last best child."
        elif candidate_pools:
            user_input += " and top-performing designs in similar datasets."
        else:
            user_input += "."
        user_input += "You shouldn’t repropose model designs that have already been validated in the optimization " \
                      "trajectory. "

        fields = {}
        for i in range(num_children):
            fields[f"{dataset_name}_refined_operation_{i}"] = (Optional[conlist(str, min_items=4, max_items=4)],
                                                               Field(default=None,
                                                                     description=f"The operation list of the children "
                                                                                 f"{i+1} suggested for the unseen "
                                                                                 f"dataset {dataset_name} as the next "
                                                                                 f"trail."))
            fields[f"{dataset_name}_refined_macro_{i}"] = (Optional[conlist(int, min_items=4, max_items=4)],
                                                           Field(default=None,
                                                                 description=f"The macro architecture list of the "
                                                                             f"children {i+1} suggested for the unseen "
                                                                             f"dataset {dataset_name} as the next "
                                                                             f"trail."))
            fields[f"{dataset_name}_refined_design_reason_{i}"] = (Optional[str],
                                                                   Field(default=None,
                                                                         description=f"Reason for the children {i+1} "
                                                                                     f"suggested for the unseen "
                                                                                     f"dataset {dataset_name} as the "
                                                                                     f"next trail."))
        optimization_tool = create_model('RefinedModelDesign', **fields)
        optimization_tool.__doc__ = f"Suggest {num_children} better GNN model architectures on the unseen dataset " \
                                    f"based on the best children from the last generation, optimization trajectory, " \
                                    f"and top-performing GNN model architectures on the most similar benchmark dataset."

        return prompt, user_input, optimization_tool

    def generate_design_evolution_prompt(self, dataset_name, num_children, generation, gnas_history,
                                         best_design, detailed_log=False, candidate_pools=None):
        """
        Generate a new LLM prompt to suggest design improvements based on performance and optionally detailed training logs.

        :param dataset_name: Unseen dataset name.
        :param current_design: Dictionary of the current design.
        :param iteration: The current iteration.
        :param gnas_history: The Graph NAS history.
        :param best_design: Dictionary of the best design.
        :param detailed_log: Use detailed training log as context or not.
        :param candidate_pools: Information about top-performing designs from similar datasets.
        :return: A string prompt for the LLM.
        """
        intro = self.generate_GNAS_task_description() + self.generate_short_space_description()

        # Building the history narrative
        history = f"Currently, you are the evolutionary Graph NAS agent at {generation} generation. We have already explored various Graph Neural Network architectures to optimize performance. Your further recommendation should not repeat any of the models in the optimization trajectory (history) below:\n"

        # Iterate over the history dictionary, sorted by iteration keys to maintain order
        for iter_num in sorted(gnas_history.keys(), key=int):
            # Access the details of each iteration
            details = gnas_history[iter_num]
            history += f"Generation {iter_num} tested {len(details)} children:\n"
            for child in details:
                history += f" - Operations {child['ops']} and macro architecture {child['link']} achieved a " \
                          f"performance of {child['perf']}.\n"

        # Highlighting the best model so far
        if best_design:
            history += f"The best model design so far is operations {best_design['ops']} and macro architecture {best_design['link']}, which achieved a performance of {round(best_design['perf'])} at generation {best_design['iteration']}.\n"


        # If detailed logs are available, add them to the prompt
        log = ""
        if detailed_log and best_design:
            log = f"Here is the training log snapshot (every 25 epochs) of the last experiment with operations {best_design['ops']} and macro architecture {best_design['link']}:\n"
            for log_entry in best_design["detailed_log"]:
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
        instruction = f"\nAs an optimal evolutionary Graph NAS, please suggest improvements or modifications to enhance " \
                  f"the model's performance for the next trial ({num_children} children models in total) based on the" \
                  f" best child from last generation, the history of experimental performances"
        if detailed_log and candidate_pools:
            instruction += ", training log of last best child, and top-performing designs in similar datasets."
        elif detailed_log:
            instruction += " and training log of last best child."
        elif candidate_pools:
            instruction += " and top-performing designs in similar datasets."
        else:
            instruction += "."
        instruction += f"The objective is to maximize the model's performance. You shouldn’t repropose model designs that " \
                  f"have already been validated in the optimization trajectory.Your need to suggest {num_children} " \
                  f"optimal model designs for the unseen dataset {dataset_name}, which should be in the same search " \
                  f"space we defined. Your answer should be in the following format:\n"
        for i in range(num_children):
            instruction += f"For ‘{dataset_name}’: (Architecture: [TBD], Operations: [TBD])\nReasons:\n"

        return intro + history + log + knowledge + instruction

    @staticmethod
    def generate_llm_selection_prompt_parser(dataset_name, children, current_design, generation, gnas_history,
                                             best_design, detailed_log, candidate_pools):
        """
        Generate a prompt for the LLM to help select the most promising child model based on the provided GNAS context.

        :param dataset_name: Name of the dataset being optimized.
        :param children: List of child models generated in the current generation.
        :param current_design: Current model design before the generation of children.
        :param generation: Current generation number in the evolutionary search.
        :param gnas_history: Historical record of all generations and their model performances.
        :param best_design: The best model design encountered so far in terms of performance.
        :param detailed_log: Flag to include detailed training logs in the prompt.
        :param candidate_pools: Information about top-performing designs from the most similar dataset.
        :return: A string prompt for the LLM.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a machine learning expert proficient in Graph Neural Networks (GNN) design and graph "
                        "dataset understanding. Your task is to perform an evolutionary neural architecture search of "
                        f"GNN on the unseen graph dataset {dataset_name} based on the optimization trajectory and "
                        "top-performing GNN model architectures on the most similar benchmark dataset.\n"
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
        user_input = f"Currently, you are the evolutionary Graph NAS agent at {generation} generation. We have " \
                     f"explored various GNN architectures to optimize performance on the unseen dataset. Here's the " \
                     f"history:\n"

        # Iterate over the history dictionary, sorted by iteration keys to maintain order
        for iter_num in sorted(gnas_history.keys(), key=int):
            # Access the details of each iteration
            details = gnas_history[iter_num]
            user_input += f" - Generation {iter_num} achieved a performance of {details['perf']} with operations " \
                          f"{details['ops']} and macro architecture {details['link']}.\n"

        # Highlighting the best model so far
        user_input += f"The best model design so far is operations {best_design['ops']} and macro architecture " \
                      f"{best_design['link']}, which achieved a performance of {best_design['perf']} at iteration " \
                      f"{best_design['iteration']}.\n"

        # Adding the performance of the most recent model
        user_input += f"The best child in last generation, which tested operations {current_design['ops']} and macro " \
                      f"architecture {current_design['link']}, achieved a performance of {current_design['perf']}.\n"

        # If detailed logs are available, add them to the prompt
        if detailed_log:
            user_input += "Here is a summary of its training log over every 25 epochs:\n"
            for log_entry in current_design["detailed_log"]:
                user_input += (f"Epoch {log_entry['epoch']}: Train Acc: {log_entry['train_accuracy']}, "
                               f"Val Acc: {log_entry['val_accuracy']}, Test Acc: {log_entry['test_accuracy']}, "
                               f"Train Loss: {log_entry['train_loss']}, Val Loss: {log_entry['val_loss']}, "
                               f"Test Loss: {log_entry['test_loss']}, Latency: {log_entry['latency']}s;\n")

        # Include insights from candidate pools if available
        if candidate_pools:
            user_input += "\nAdditionally, please consider insights from top-performing designs in similar benchmark " \
                          "datasets: \n"
            for pool_key, pool in candidate_pools.items():
                for dataset_info in pool:
                    selected_dataset = dataset_info['selected_dataset']
                    top_models = dataset_info['top_models']
                    user_input += f"From similar dataset '{selected_dataset}', top model designs include:\n"
                    for model in top_models:
                        architecture, operations = model
                        user_input += f"  - Architecture: {architecture}, Operations: {operations}\n"

        # Children details for selection
        # prompt += "We have completed the crossover on the best child from the last generation with respect to the top " \
        #         "model designs from the second and third similar datasets. Here is the current generation of " \
        #         "children for selection:\n"
        user_input += "We have completed the crossover on the best design so far with respect to the top model " \
                      "designs from the three benchmark datasets with the highest similarity. Here is the current " \
                      "generation of children for your selection:\n"
        for idx, child in enumerate(children):
            user_input += f"Child {idx + 1}: Architecture {child['link']}, Operations {child['ops']}\n"

        # Finally, ask for suggestions on improvements
        user_input += "\nYour objective is to maximize the model's performance on the unseen dataset. As an optimal " \
                  "Graph NAS and mutation operator, please suggest the best child from the current generation and " \
                  "perform evolutionary mutation on it for the next trial based on the potential effectiveness of " \
                  "these children, the history of experimental performances"
        # prompt += "\nAs an optimal Graph NAS, please suggest the best child from the current generation for further " \
        #          "validation based on their potential effectiveness, the history of experimental performances"
        if detailed_log and candidate_pools:
            user_input += ", training log of last trial, and top-performing designs in similar benchmark datasets."
        elif detailed_log:
            user_input += " and training log of last trial."
        elif candidate_pools:
            user_input += " and top-performing designs in similar benchmark datasets."
        else:
            user_input += "."
        user_input += "You shouldn’t repropose model designs that have already been validated in the optimization " \
                      "trajectory. "

        fields = {}
        fields[f"{dataset_name}_refined_operation"] = (Optional[conlist(str, min_items=4, max_items=4)],
                                                       Field(default=None,
                                                             description=f"The operation list of the mutated model "
                                                                         f"design suggested for the unseen dataset "
                                                                         f"{dataset_name} as the next trail."))
        fields[f"{dataset_name}_refined_macro"] = (Optional[conlist(int, min_items=4, max_items=4)],
                                                   Field(default=None,
                                                         description=f"The macro architecture list of the mutated "
                                                                     f"model design suggested for the unseen dataset "
                                                                     f"{dataset_name} as the next trail."))
        fields[f"{dataset_name}_refined_design_reason"] = (Optional[str],
                                                           Field(default=None,
                                                                 description=f"Reason for the mutated model design "
                                                                             f"suggested for the unseen dataset "
                                                                             f"{dataset_name} as the next trail."))
        optimization_tool = create_model('MutatedModelDesign', **fields)
        optimization_tool.__doc__ = "Suggest a better GNN model architecture on the unseen dataset based on the " \
                                    "potentially best children from the current generation, optimization trajectory, " \
                                    "and top-performing GNN model architectures on the most similar benchmark dataset."

        return prompt, user_input, optimization_tool

    @staticmethod
    def generate_llm_selection_prompt(dataset_name, children, current_design, generation, gnas_history, best_design,
                                      detailed_log, candidate_pools):
        """
        Generate a prompt for the LLM to help select the most promising child model based on the provided GNAS context.

        :param dataset_name: Name of the dataset being optimized.
        :param children: List of child models generated in the current generation.
        :param current_design: Current model design before the generation of children.
        :param generation: Current generation number in the evolutionary search.
        :param gnas_history: Historical record of all generations and their model performances.
        :param best_design: The best model design encountered so far in terms of performance.
        :param detailed_log: Flag to include detailed training logs in the prompt.
        :param candidate_pools: Information about top-performing designs from the most similar dataset.
        :return: A string prompt for the LLM.
        """
        intro = (f"You are an expert in the field of neural architecture search. Your task is to perform neural "
                 f"architecture search of Graph Neural Network on the unseen graph dataset {dataset_name}. To recall, "
                 f"in the context of GNN, the design of a model is described by two main components:\n"
                 f"1. The macro architecture is represented as a directed acyclic graph (DAG), dictating the flow of "
                 f"data through various operations. The macro space can be described by a list of integers, indicating "
                 f"the input node index for each computing node (0 for the raw input, 1 for the first computing node, "
                 f"etc.) We consider 9 distinct DAG configurations in our search space: [0, 0, 0, 0], [0, 0, 0, 1], "
                 f"[0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3], [0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 2], [0, 1, 2, 3].\n"
                 f"2. The operations applied at each node, specified by a list of strings. We consider 9 candidate "
                 f"operations, which are: 'gat', 'gcn', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip'. \n"
                 f"Together, these components define the computation graph of a GNN, including the flow of data through"
                 f" various operations. The meaning behind each component of this search space has been introduced "
                 f"before.\n")

        # Building the history narrative
        prompt = f"Currently, you are the evolutionary Graph NAS agent at {generation} generation. We have explored " \
                 f"various Graph Neural Network architectures to optimize performance. Here's the history:\n"

        # Iterate over the history dictionary, sorted by iteration keys to maintain order
        for iter_num in sorted(gnas_history.keys(), key=int):
            # Access the details of each iteration
            details = gnas_history[iter_num]
            prompt += f" - Generation {iter_num} achieved a performance of {details['perf']} with operations " \
                      f"{details['ops']} and macro architecture {details['link']}.\n"

        # Highlighting the best model so far
        prompt += f"The best model design so far is operations {best_design['ops']} and macro architecture " \
                  f"{best_design['link']}, which achieved a performance of {best_design['perf']} at generation " \
                  f"{best_design['iteration']}.\n"

        # Adding the performance of the most recent model
        prompt += f"The most recent model design, which tested operations {current_design['ops']} and macro " \
                  f"architecture {current_design['link']}, achieved a performance of {current_design['perf']}.\n"

        # If detailed logs are available, add them to the prompt
        if detailed_log:
            prompt += "Here is a summary of the training log over every 25 epochs:\n"
            for log_entry in current_design["detailed_log"]:
                prompt += (f"Epoch {log_entry['epoch']}: Train Acc: {log_entry['train_accuracy']}, "
                           f"Val Acc: {log_entry['val_accuracy']}, Test Acc: {log_entry['test_accuracy']}, "
                           f"Train Loss: {log_entry['train_loss']}, Val Loss: {log_entry['val_loss']}, "
                           f"Test Loss: {log_entry['test_loss']}, Latency: {log_entry['latency']}s;\n")

        # Include insights from candidate pools if available
        if candidate_pools:
            prompt += "\nAdditionally, insights from top-performing designs in similar datasets include:\n"
            top_models = candidate_pools[dataset_name][0]
            selected_dataset = top_models['selected_dataset']
            top_models = top_models['top_models']
            prompt += f"From similar dataset '{selected_dataset}', top model designs include:\n"
            for model in top_models:
                prompt += f" - Architecture: {model[0]}, Operations: {model[1]}\n"

        # Children details for selection
        #prompt += "We have completed the crossover on the best child from the last generation with respect to the top " \
        #         "model designs from the second and third similar datasets. Here is the current generation of " \
        #         "children for selection:\n"
        prompt += "We have completed the crossover on the best design so far with respect to the top model designs " \
                  "from the first and second similar datasets. Here is the current generation of " \
                  "children for selection:\n"
        for idx, child in enumerate(children):
            prompt += f"Child {idx + 1}: Architecture {child['link']}, Operations {child['ops']}\n"

        # Finally, ask for suggestions on improvements
        prompt += "\nAs an optimal Graph NAS and mutation operator, please suggest the best child from the current " \
                  "generation and perform evolutionary mutation on it for further validation based on their potential" \
                  " effectiveness, the history of experimental performances"
        #prompt += "\nAs an optimal Graph NAS, please suggest the best child from the current generation for further " \
        #          "validation based on their potential effectiveness, the history of experimental performances"
        if detailed_log and candidate_pools:
            prompt += ", training log of last trial, and top-performing designs in the most similar datasets."
        elif detailed_log:
            prompt += " and training log of last trial."
        elif candidate_pools:
            prompt += " and top-performing designs in the most similar datasets."
        else:
            prompt += "."
        prompt += f"The objective is to maximize the model's performance. You shouldn’t repropose model designs that " \
                  f"have already been validated in the optimization trajectory.Your suggested optimal model design " \
                  f"for the unseen dataset {dataset_name} should be in the same search space we defined. Your answer " \
                  f"should be in the following format:\n"
        prompt += f"For ‘{dataset_name}’: (Architecture: [TBD], Operations: [TBD])\nReasons:\n"

        return intro + prompt
    
    @staticmethod
    def generate_GPT4GNAS_prompt(num_children, generation, gnas_history):
        """
        Generate a new LLM prompt to suggest design improvements based on performance and optionally detailed training logs.

        :param dataset_name: Unseen dataset name.
        :param current_design: Dictionary of the current design.
        :param iteration: The current iteration.
        :param gnas_history: The Graph NAS history.
        :param best_design: Dictionary of the best design.
        :param detailed_log: Use detailed training log as context or not.
        :param candidate_pools: Information about top-performing designs from similar datasets.
        :return: A string prompt for the LLM.
        """
        prompt = f"// Search Task\nThe task is to choose the best GNN architecture on a given dataset. The architecture will be trained and tested on the unseen dataset, and the objective is to maximize model accuracy.\n// Search Space\nA GNN architecture is defined as follows: The first operation is input, the last operation is output, and the intermediate operations are candidate operations. The adjacency matrix of operation connections is as follows: [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3], [0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 2], [0, 1, 2, 3], where the (i,j)-th element in the adjacency matrix denotes that the output of operation i will be used as the input of operation j. There are [Candidate Numbers] operations that can be selected: ['gat', 'gcn', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip'].\n// Search Strategy\nAt the beginning, when only a few numbers of evaluated architectures are available, use the exploration strategy to explore the operations. Randomly select a batch of operations for evaluation. When a certain amount of evaluated architectures are available, use the exploitation strategy to find the best operations by sampling the best candidate operations from previously generated candidates.\n"

        # Building the history narrative
        if generation == 0:
            prompt += f"Currently, you are at generation {generation}. We have not yet explored any GNN architectures.\n"
        else:
            prompt = f"Currently, you are at generation {generation}. We have explored various Graph Neural Network architectures to optimize performance. Here's the history:\n"
            # Iterate over the history dictionary, sorted by iteration keys to maintain order
            for iter_num in sorted(gnas_history.keys(), key=int):
                # Access the details of each iteration
                details = gnas_history[iter_num]
                prompt += f"Generation {iter_num} tested {len(details)} children:\n"
                for child in details:
                    prompt += f" - Operations {child['ops']} and macro architecture {child['link']} achieved a " \
                            f"performance of {child['perf']}.\n"

        # Finally, ask for suggestions on improvements
        prompt += f"You need to suggest {num_children} non-repeating model designs for the unseen dataset, which should be in the same search space we defined. Please remember that your suggested architectures can only be any of the following: [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3], [0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 2], [0, 1, 2, 3], and your suggested operations can only be any 4-elements combination of operations in ['gat', 'gcn', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip'] (individual operation can be repeated). \nYour answer should closely follow the output format below:\n\nResponse Format:\n"
        for i in range(num_children):
            prompt += f"{i+1}. (Architecture: [X, X, X, X], Operations: [X, X, X, X])\n"

        return prompt
    
    @staticmethod
    def generate_GHGNAS_prompt(dataset, num_children, generation, gnas_history):
        """
        Generate a new LLM prompt to suggest design improvements based on performance and optionally detailed training logs.

        :param dataset_name: Unseen dataset name.
        :param current_design: Dictionary of the current design.
        :param iteration: The current iteration.
        :param gnas_history: The Graph NAS history.
        :param best_design: Dictionary of the best design.
        :param detailed_log: Use detailed training log as context or not.
        :param candidate_pools: Information about top-performing designs from similar datasets.
        :return: A string prompt for the LLM.
        """
        prompt = "Our task is graph neural architecture, searching for a GNN architecture that can achieve the best performance on a downstream task."
        if dataset == 'Planetoid:Cora':
            prompt += "The Cora dataset is a citation graph where vertices represent papers and links represent citations between papers. Features are bag-of-words and labels are ground-truth topics. In summary, its statistics contain 1433 features, 7 single classes, and accuracy as the evaluation metric. "
        elif dataset == 'Planetoid:CiteSeer':
            prompt += "The Citeseer dataset is a citation graph where vertices represent papers and links represent citations between papers. Features are bag-of-words and labels are ground-truth topics. In summary, its statistics 3703 features, 6 single classes, and accuracy as the evaluation metric. "
        elif dataset == 'Planetoid:PubMed':
            prompt += "The Pubmed dataset is a citation graph where vertices represent papers and links represent citations between papers. Features are bag-of-words and labels are ground-truth topics. In summary, its statistics contain 500 features, 3 single classes, and accuracy as the evaluation metric. "
        elif dataset == 'Coauthor:CS':
            prompt += "The Coauthor CS dataset is a co-authorship graph from Microsoft Academic Graph. Vertices represent authors, links represent co-author relationships, features represent the authors' paper keywords, and vertices labels indicate the author's research fields. In summary, its statistics contain 6805 features, 15 single classes, and accuracy as the evaluation metric. "
        elif dataset == 'Coauthor:Physics':
            prompt += "The Coauthor Physics dataset is a co-authorship graph from Microsoft Academic Graph. Vertices represent authors, links represent co-author relationships, features represent the authors' paper keywords, and vertices labels indicate the author's research fields. In summary, its statistics contain 8415 features, 5 single classes, and accuracy as the evaluation metric. "
        elif dataset == 'Amazon:Photo':
            prompt += "The Amazon Photo dataset is a subset of Amazon's co-purchase graph. Vertices represent products, and links between products represent that they are frequently bought together, features are bag-of-words of product reviews, and vertices labels are the product category. In summary, its statistics contain 745 features, 8 single classes, and accuracy as the evaluation metric. "
        elif dataset == 'Amazon:Computers':
            prompt += "The Amazon Computers dataset is a subset of Amazon's co-purchase graph. Vertices represent products, and links between products represent that they are frequently bought together, features are bag-of-words of product reviews, and vertices labels are the product category. In summary, its statistics contain 767 features, 10 single classes, and accuracy as the evaluation metric. "
        elif dataset == 'ogbn-arxiv':
            prompt += "The ogbn-arXiv dataset, a part of the Open Graph Benchmark, is a graph representing the citation relationships between papers from the arXiv Computer Science (CS) category indexed by MAG. Each paper has a feature vector based on word embedding in its title and abstract. Labels indicate the subject areas of papers, and the dataset is split based on chronological order. In summary, its statistics contain 128 features, 40 single classes, and accuracy as the evaluation metric. "
        elif dataset == 'CitationFull:DBLP':
            prompt += "The unseen dataset is a citation network dataset. The citation data is extracted from DBLP, ACM, MAG (Microsoft Academic Graph), and other sources. Each paper is associated with abstract, authors, year, venue, and title. The data set can be used for clustering with network and side information, studying influence in the citation network, finding the most influential papers, topic modeling analysis, etc. In summary, its statistics contain 1,639 features, 4 single classes, and accuracy as the evaluation metric."
        elif dataset == 'Flickr':
            prompt += "The unseen dataset originates from NUS-wide. This dataset is built by forming links between images sharing common metadata from Flickr. Edges are formed between images from the same location, submitted to the same gallery, group, or set, images sharing common tags, images taken by friends, etc. The original images are collected from PASCAL, ImageCLEF, MIR, and NUS-wide. It generated an undirected graph. One node in the graph represents one image uploaded to Flickr. If two images share some common properties (e.g., same geographic location, same gallery, comments by the same user, etc.), there is an edge between the nodes of these two images. We use the 500-dimensional bag-of-word representation of the images provided by NUS-wide as the node features. For labels, we scanned over the 81 tags of each image and manually merged them into 7 classes. Each image belongs to one of the 7 classes. In summary, its statistics contain 500 features, and 7 single classes."
        elif dataset == 'Actor':
            prompt += "This dataset is the actor-only induced subgraph of the film-director-actor-writer network. Each nodes correspond to an actor, and the edge between two nodes denotes co-occurrence on the same Wikipedia page. Node features correspond to some keywords in the Wikipedia pages. In summary, its statistics 932 features, 5 single classes, and accuracy as the evaluation metric. "
        else:
            raise NotImplementedError(f"Dataset {dataset} is not supported.")

        prompt += f"A GNN architecture is defined as follows: The first operation is input, the last operation is output, and the intermediate operations are candidate operations. The adjacency matrix of operation connections is as follows: [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3], [0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 2], [0, 1, 2, 3], where the (i,j)-th element in the adjacency matrix denotes that the output of operation i will be used as the input of operation j. There are [Candidate Numbers] operations that can be selected: ['gat', 'gcn', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip'].\nExploration Strategy: Explore as many different architectures in the search space as possible. Optimization Strategy: Analyze how to get a better architecture based on existing results.\n"

        # Building the history narrative
        if generation == 0:
            prompt += f"Currently, you are at generation {generation}. We have not yet explored any GNN architectures.\n"
        else:
            prompt = f"Currently, you are at generation {generation}. We have explored various Graph Neural Network architectures to optimize performance. Here's the history:\n"
            # Iterate over the history dictionary, sorted by iteration keys to maintain order
            for iter_num in sorted(gnas_history.keys(), key=int):
                # Access the details of each iteration
                details = gnas_history[iter_num]
                prompt += f"Generation {iter_num} tested {len(details)} children:\n"
                for child in details:
                    prompt += f" - Operations {child['ops']} and macro architecture {child['link']} achieved a " \
                            f"performance of {child['perf']}.\n"

        # Finally, ask for suggestions on improvements
        prompt += f"You need to suggest {num_children} non-repeating model designs for the unseen dataset, which should be in the same search space we defined. Please remember that your suggested architectures can only be any of the following: [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3], [0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 2], [0, 1, 2, 3], and your suggested operations can only be any 4-elements combination of operations in ['gat', 'gcn', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip'] (individual operation can be repeated). \nYour answer should closely follow the output format below:\n\nResponse Format:\n"
        for i in range(num_children):
            prompt += f"{i+1}. (Architecture: [X, X, X, X], Operations: [X, X, X, X])\n"

        return prompt
    
    def generate_llm_prompt(self, dataset_name, current_design, generation, gnas_history, best_design, detailed_log, 
                            candidate_pools):
        """
        Generate a prompt for the LLM to help refine the promoted child model based on the provided GNAS context.

        :param dataset_name: Name of the dataset being optimized.
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

        # Finally, ask for suggestions on improvements
        instruction = f"\nAs an optimal Graph NAS that performs exploitation (mutation), please further refine (mutate) the best child so far based on the experiment history"
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
        instruction += "The objective is to maximize the model's performance. You shouldn't repropose model designs that have already been validated in the optimization trajectory. Your suggested optimal model design for the unseen dataset should be in the same search space we defined and should not repeat any model design already contained in the experiment history. Your answer should closely follow the output format below:\n\n"
        instruction += f"Response Format:\nFor the unseen dataset, I recommend (Architecture: [TBD], Operations: [TBD]).\nReasons for recommendation: TBD\n"

        return intro + history + log + knowledge + instruction
    
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
    
    def generate_kg_prompt(self, current_design, generation, gnas_history, best_design, detailed_log, candidate_pools):
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
            knowledge = f"\nAdditionally, the candidate models below are similar to the current model in structure and hyper-parameters, and have close performances on the most similar benchmark datasets:\n"
            for model in candidate_pools:
                knowledge += f" - Architecture: {model[0]}, Operations: {model[1]}\n"
        else:
            knowledge = ""

        # Finally, ask for suggestions on improvements
        #instruction = f"\nAs an optimal Graph NAS that performs exploitation (mutation), please further refine (mutate) this current child ({current_design['link']}, {current_design['ops']}) based on its potential effectiveness, the experiment history"
        instruction = f"\nAs an optimal Graph NAS, please select a new candidate from the candidate model list above based on their potential effectiveness, the experiment history"
        #prompt += "\nAs an optimal Graph NAS, please suggest the best child from the current generation for further " \
        #          "validation based on their potential effectiveness, the history of experimental performances"
        if detailed_log and candidate_pools:
            instruction += ", training log of last trial, and the potential pattern of similar models."
        elif detailed_log:
            instruction += " and training log of last trial."
        elif candidate_pools:
            instruction += " and the potential pattern of similar models."
        else:
            instruction += "."
        #instruction += "The objective is to maximize the model's performance. You should modify upon the current child and shouldn't repropose model designs that have already been validated in the optimization trajectory. Your suggested optimal model design for the unseen dataset should be in the same search space we defined and should not repeat any model design already contained in the experiment history. Your answer should closely follow the output format below:\n\n"
        instruction += "The objective is to maximize the model's performance. You should select from the given candidate model list and shouldn't repropose model designs that have already been validated in the optimization trajectory. Your suggested optimal model design for the unseen dataset should be in the same search space we defined and should not repeat any model design already contained in the experiment history. Your answer should closely follow the output format below:\n\n"
        instruction += f"Response Format:\nFor the unseen dataset, I recommend (Architecture: [TBD], Operations: [TBD]).\nReasons for recommendation: TBD\n"

        return intro + history + log + knowledge + instruction

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
                "link": ast.literal_eval(architecture),
                "ops": ast.literal_eval(operations)
            }

        return suggested_designs

    @staticmethod
    def extract_model_designs_evolution(llm_response):
        """
        Extracts model designs suggested by the LLM for each source dataset.

        :param llm_response: A string containing the LLM's response in the specified format.
        :return: A list of suggested model designs.
        """
        # Pattern to match the format of the LLM's response for each dataset
        pattern = r"\(Architecture: (\[.*?\]), Operations: (\[.*?\])\)"

        # Find all matches in the response
        matches = re.findall(pattern, llm_response)

        # Initialize a dictionary to hold the extracted designs
        suggested_designs = []

        # Iterate through all matches and populate the dictionary
        for match in matches:
            architecture, operations = match
            suggested_designs.append({
                "link": ast.literal_eval(architecture),
                "ops": ast.literal_eval(operations)
            })

        return suggested_designs

    @staticmethod
    def describe_model_design(link_structure, operations):
        """
        Generates a textual description of a GNN architecture based on its design components,
        detailing the sequence and concatenation of operations as specified.

        :param link_structure: A list of integers indicating the input node index for each computing node.
        :param operations: A list of strings representing the operations performed at each node.
        :return: A textual description of the GNN architecture.
        """
        operation_names = {
            'gcn': 'Graph Convolutional Network layer',
            'gat': 'Graph Attention Network layer',
            'gin': 'Graph Isomorphism Network layer',
            'cheb': 'Chebyshev Spectral Graph Convolution',
            'sage': 'GraphSAGE',
            'arma': 'ARMA layer',
            'graph': 'k-GNN',
            'fc': 'Fully Connected layer',
            'skip': 'Skip Connection'
        }

        paths = {}  # Tracks the paths from input to each node
        for index, (input_index, op) in enumerate(zip(link_structure, operations)):
            op_name = operation_names[op]
            if input_index == 0:  # Directly from Input
                paths[index] = [f"Input -> {op_name}"]
            else:  # From another node
                paths[index] = [f"{paths[input_index - 1][-1]} -> {op_name}"]

        # Determining nodes without successors (for concatenation)
        outputs = []
        for i in range(len(operations)):
            if i + 1 not in link_structure:  # This node's output is not an input to any other node
                outputs.append(operations[i])

        # Creating the final output description
        if len(outputs) > 1:  # Concatenation case
            output_desc = "[" + ", ".join([operation_names[op] for op in outputs]) + "] -> Output"
        else:  # Single output case
            output_desc = paths[len(operations) - 1][-1] + " -> Output"

        # Combine all paths to a final description
        description = ". ".join([path[-1] for path in paths.values()]) + ". " + output_desc

        return description

    @staticmethod
    def write_design_report(data, filename):
        """
        Writes the structured response containing design operations, macro architectures, and reasons into a .txt file.

        :param data: Dictionary containing the structured response.
        :param filename: The name of the file to write to.
        """
        with open(filename, 'w') as file:
            # Iterate over the keys to process each dataset section
            for key in data:
                if key.endswith('initial_operation'):
                    operations = data[key]
                    macros = data[f"initial_macro"]
                    reason = data[f"initial_design_reason"]

                    # Write the formatted output to the file
                    file.write("Operations: " + ', '.join(operations) + "\n")
                    file.write("Macro Architecture: " + ', '.join(map(str, macros)) + "\n")
                    file.write("Design Reason:\n" + reason + "\n")
                    file.write("\n")

    @staticmethod
    def write_optimization_report(data, filename, iteration):
        """
        Writes the structured response containing design operations, macro architectures, and reasons into a .txt file.

        :param data: Dictionary containing the structured response.
        :param filename: The name of the file to write to.
        """
        with open(filename, 'a') as file:
            # Iterate over the keys to process each dataset section
            for key in data:
                if key.endswith('_refined_operation'):
                    dataset_name = key.replace('_refined_operation', '')
                    operations = data[key]
                    macros = data[f"{dataset_name}_refined_macro"]
                    reason = data[f"{dataset_name}_refined_design_reason"]

                    # Write the formatted output to the file
                    file.write(f"Response for iteration {iteration}:\n")
                    file.write("Operations: " + ', '.join(operations) + "\n")
                    file.write("Macro Architecture: " + ', '.join(map(str, macros)) + "\n")
                    file.write("Design Reason:\n" + reason + "\n")
                    file.write("\n")

    @staticmethod
    def write_evolutionary_report(data, filename, generation, dataset_name, num_children):
        """
        Writes the structured response containing design operations, macro architectures, and reasons into a .txt file.

        :param data: Dictionary containing the structured response.
        :param filename: The name of the file to write to.
        """
        with open(filename, 'a') as file:
            file.write(f"Response for iteration {generation}:\n")
            # Iterate over the keys to process each dataset section
            for i in range(num_children):
                # Extract the keys based on the dataset name
                operations = data[f"{dataset_name}_refined_operation_{i}"]
                macros = data[f"{dataset_name}_refined_macro_{i}"]
                reason = data[f"{dataset_name}_refined_design_reason_{i}"]

                # Write the formatted output to the file
                file.write(f"Children {i + 1}:\n")
                file.write("Operations: " + ', '.join(operations) + "\n")
                file.write("Macro Architecture: " + ', '.join(map(str, macros)) + "\n")
                file.write("Design Reason:\n" + reason + "\n")
            file.write("\n")

    @staticmethod
    def reformat_suggested_design(suggested_design, dataset_name):
        # Initialize the new dictionary
        suggested_design_dict = {}

        # Extract the keys based on the dataset name
        ops_key = f"initial_operation"
        macro_key = f"initial_macro"

        # Populate the new dictionary with structured data
        suggested_design_dict[dataset_name] = {
            "link": suggested_design.get(macro_key, []),
            "ops": suggested_design.get(ops_key, []),
        }

        return suggested_design_dict

    @staticmethod
    def reformat_refined_design(suggested_design, dataset_name):
        # Initialize the new dictionary
        suggested_design_dict = {}

        # Extract the keys based on the dataset name
        ops_key = f"{dataset_name}_refined_operation"
        macro_key = f"{dataset_name}_refined_macro"

        # Populate the new dictionary with structured data
        suggested_design_dict[dataset_name] = {
            "link": suggested_design.get(macro_key, []),
            "ops": suggested_design.get(ops_key, []),
        }

        return suggested_design_dict

    @staticmethod
    def reformat_evolutionary_design(suggested_design, dataset_name, num_children):
        # Initialize the new dictionary
        suggested_designs = []

        for i in range(num_children):
            # Extract the keys based on the dataset name
            ops_key = f"{dataset_name}_refined_operation_{i}"
            macro_key = f"{dataset_name}_refined_macro_{i}"

            suggested_designs.append({
                "link": suggested_design.get(macro_key, []),
                "ops": suggested_design.get(ops_key, []),
            })

        return suggested_designs

    @staticmethod
    def has_bad_models(models_info):
        for model in models_info:
            if 'bad_models' in model and model['bad_models']:
                return True
        return False

