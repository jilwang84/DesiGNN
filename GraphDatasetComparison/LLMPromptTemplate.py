# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, create_model


class LLMPromptTemplate:
    
    @staticmethod
    def generate_task_similarity_prompt_parser_metric(unseen_dataset_description, benchmark_dataset_descriptions,
                                                      similarity_scores=None, use_semantic=False):
        """
        Generates a prompt that combines textual and mathematical similarities.
        :param unseen_dataset_description: Description of the unseen dataset.
        :param benchmark_dataset_descriptions: Dictionary of descriptions for public datasets.
        :param similarity_scores: Dictionary of similarity scores between the unseen dataset and public datasets.
        :return: A formatted LLM prompt string.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a machine learning expert proficient in graph neural network design and graph dataset "
                        "understanding. You will analyze an unseen dataset provided by the user and multiple benchmark "
                        "datasets to measure the similarity between the unseen dataset and other benchmark datasets in "
                        "the optimal GNN model architecture. Your ultimate goal is to recommend a GNN model "
                        "architecture that performs well on the unseen dataset based on the best GNN model "
                        "architectures on similar benchmark datasets to the user.\nFor your information, a classical "
                        "task similarity metric in graph neural networks is measured by how different GNN "
                        "architectures perform across various datasets. A set of diverse GNN designs, known as anchor "
                        "models, are applied to these tasks to capture different aspects. We can quantify task "
                        "similarity by comparing the performance rankings of these models across datasets using the "
                        "Kendall rank correlation. This metric helps transfer the best GNN architectures or "
                        "principles across similar datasets."),
             ("user", "{input}")]
        )
        user_input = "Based on the given descriptions of the unseen dataset and multiple benchmark datasets below, please take a deep breath and work on this problem step-by-step: analyze the characteristics of datasets that affect their choice of best GNN architectures in practice and estimate the task similarity scores between the unseen dataset and each benchmark dataset. Please pay special attention to the similarities between the metrics listed in each dataset description, which are empirically found to be consistent with good GNN models across different datasets. The task similarity score should range from -1 (completely dissimilar tasks) to 1 (identical tasks). Here are the descriptions:\n"
        user_input += "Unseen Dataset's Description:\n"
        user_input += f"{unseen_dataset_description}\n"

        user_input += "Benchmark Datasets' Descriptions:\n"
        i = 1
        fields = {}
        for dataset_name, description in benchmark_dataset_descriptions.items():
            if ':' in dataset_name:
                dataset_name = dataset_name.split(':')[1]
            fields[f"{dataset_name}_similarity_score"] = (Optional[float], Field(default=None,
                                                                                 description=f"Task similarity between "
                                                                                             f"the unseen dataset and "
                                                                                             f"the benchmark dataset "
                                                                                             f"{dataset_name}."))
            fields[f"{dataset_name}_reason"] = (Optional[str], Field(default=None,
                                                                     description=f"Reason for similarity score between "
                                                                                 f"the unseen dataset and the "
                                                                                 f"benchmark dataset {dataset_name}."))

            user_input += f"({i}) Benchmark dataset {dataset_name}: {description}\n"
            if similarity_scores:
                user_input += f" (Mathematical Similarity Score): {similarity_scores.get(dataset_name, 'N/A')}\n"
            i += 1
        similarity_tool = create_model('SimilarityScores', **fields)
        similarity_tool.__doc__ = "Estimate and justify the task similarity scores between the unseen dataset and " \
                                  "each benchmark dataset based on the dataset statistics."
        
        return prompt, user_input, similarity_tool

    @staticmethod
    def generate_task_similarity_prompt_metric(unseen_dataset_description, benchmark_dataset_descriptions,
                                               similarity_scores=None):
        """
        Generates a prompt that combines textual and mathematical similarities.
        :param unseen_dataset_description: Description of the unseen dataset.
        :param benchmark_dataset_descriptions: Dictionary of descriptions for public datasets.
        :param similarity_scores: Dictionary of similarity scores between the unseen dataset and public datasets.
        :return: A formatted LLM prompt string.
        """
        # Summary of the task similarity metric concept
        metric_summary = "Task similarity in graph neural networks is measured based on how different GNN designs " \
                         "perform across various tasks. A set of diverse GNN designs, known as anchor models, are " \
                         "applied to these tasks to capture different aspects. By comparing the performance rankings " \
                         "of these models across tasks using the Kendall rank correlation, we can quantify task " \
                         "similarity. This metric helps in transferring the best GNN designs or principles across " \
                         "similar tasks.\n\n"

        # Start with the metric summary
        prompt = metric_summary

        # Introduction to the LLM task
        prompt += "Based on the summary above and the given descriptions of various graph datasets below, evaluate and provide the task similarity scores between the unseen dataset and each of the benchmark datasets. Please pay special attention to the similarities between the metrics listed in each dataset description, which are empirically found to be consistent with good GNN models across different datasets.\n\n"

        prompt += "Unseen Dataset's Description:\n"
        prompt += f"{unseen_dataset_description}\n"

        prompt += "Benchmark Datasets' Descriptions:\n"
        for dataset_name, description in benchmark_dataset_descriptions.items():
            prompt += f"{description}\n"
            if similarity_scores:
                prompt += f" (Mathematical Similarity Score): {similarity_scores.get(dataset_name, 'N/A')}\n"

        # Instruct the LLM on the expected output format
        prompt += "For each pair of unseen dataset and benchmark dataset, provide a task similarity score ranging " \
                  "from -1 (completely dissimilar tasks) to 1 (identical tasks). Please format your response as a " \
                  "list of dataset pairs with their corresponding similarity scores, along with brief justifications " \
                  "for each score based on the dataset summary and the task similarity metric concept. For example:\n"
        prompt += "- Unseen Dataset and Benchmark Dataset 1: 0.75\n"
        prompt += "- Unseen Dataset and Benchmark Dataset 2: -0.60\n"
        prompt += "...\n"
        prompt += "- Unseen Dataset and Benchmark Dataset 9: 0.50\n"

        print(prompt)

        return prompt

    @staticmethod
    def generate_self_evaluation_prompt(benchmark_dataset_descriptions, similarity_scores=None):
        """
        Generates a prompt that combines textual and mathematical similarities.
        :param benchmark_dataset_descriptions: Dictionary of descriptions for public datasets.
        :param similarity_scores: Dictionary of similarity scores between the unseen dataset and public datasets.
        :return: A formatted LLM prompt string.
        """
        # Summary of the task similarity metric concept
        metric_summary = "Task similarity in graph neural networks is measured based on how different GNN designs " \
                         "perform across various tasks. A set of diverse GNN designs, known as anchor models, are " \
                         "applied to these tasks to capture different aspects. By comparing the performance rankings " \
                         "of these models across tasks using the Kendall rank correlation, we can quantify task " \
                         "similarity. This metric helps in transferring the best GNN designs or principles across " \
                         "similar tasks.\n\n"

        # Start with the metric summary
        prompt = metric_summary

        # Introduction to the LLM task
        prompt += "Based on the summary above and the given descriptions of various graph datasets below, evaluate " \
                  "and provide the task similarity scores between each pair of datasets, whether it is an unseen " \
                  "dataset or a benchmark dataset. Consider the characteristics such as graph size, density, types " \
                  "of node and edge attributes, and the nature of prediction tasks (e.g., node classification, " \
                  "link prediction) described in each dataset summary or in your own knowledge base.\n\n"

        prompt += "All Datasets' Descriptions:\n"

        for dataset_name, description in benchmark_dataset_descriptions.items():
            prompt += f"{description}\n"
            if similarity_scores:
                prompt += f" (Mathematical Similarity Score): {similarity_scores.get(dataset_name, 'N/A')}\n"

        # Instruct the LLM on the expected output format
        prompt += "For each pair of datasets, provide a task similarity score ranging from -1 (completely " \
                  "dissimilar tasks) to 1 (identical tasks). Please format your response as a list of dataset " \
                  "pairs with their corresponding similarity scores, along with brief justifications for each score " \
                  "based on the dataset descriptions and the task similarity metric concept. For example:\n"
        prompt += "- Dataset 1 and Dataset 2: 0.75\n"
        prompt += "- Dataset 1 and Dataset 3: -0.60\n"
        # And so on for all combinations...
        prompt += "...\n"
        prompt += "- Dataset 8 and Dataset 9: 0.50\n"

        return prompt

    