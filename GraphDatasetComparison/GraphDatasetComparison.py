# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

import os
import pandas as pd
import numpy as np
from langchain_core.utils.function_calling import convert_to_openai_function

from .GraphSimilarityCalculator import GraphSimilarityCalculator
from .LLMPromptTemplate import LLMPromptTemplate


class GraphDatasetComparison:
    def __init__(self, unseen_dataset_description, benchmark_dataset_descriptions, unseen_dataset_name,
                 benchmark_datasets, unseen_dir='datasets/Unseen datasets/',
                 benchmark_dir='datasets/Benchmark datasets/', use_parser=False):
        """
        Initialize the GraphDatasetComparison class.
        :param unseen_dataset_description: The textual description of the unseen dataset.
        :param benchmark_dataset_descriptions: A dictionary containing the textual descriptions of public datasets.
        :param unseen_dataset_name: Name of the unseen dataset to be loaded.
        :param benchmark_datasets: Name of the benchmark dataset to be loaded.
        :param unseen_dir: The directory where unseen dataset-related files are stored.
        :param benchmark_dir: The directory where benchmark dataset-related files are stored.
        """
        self.unseen_dataset_description = unseen_dataset_description
        self.benchmark_dataset_descriptions = benchmark_dataset_descriptions
        self.unseen_dataset_name = unseen_dataset_name
        self.benchmark_datasets = benchmark_datasets
        self.unseen_dir = unseen_dir
        self.benchmark_dir = benchmark_dir
        self.similarity_calculator = GraphSimilarityCalculator()
        self.use_parser = use_parser

    def compare_datasets(self, langchain_query, task_similarity_file_name):
        """
        Compares the unseen dataset with public datasets and identifies the top 3 similar ones.
        :param langchain_query: Function to query the LLM.
        :return: List of top 3 similar public datasets.
        """
        # Calculate mathematical similarity
        #math_similarities = self.calculate_mathematical_similarity(measures)

        # Get textual similarity via LLM
        textual_similarity_response = self.query_similarity(langchain_query, None)

        # Write the response to the file
        if self.use_parser:
            self.write_similarity_report(textual_similarity_response, task_similarity_file_name)
        else:
            with open(task_similarity_file_name, 'w') as file:
                file.write(textual_similarity_response.content + "\n")

        return textual_similarity_response

    def calculate_mathematical_similarity(self, measures):
        """
        Calculates mathematical similarity scores between the unseen dataset and public datasets.
        :param measures: List of measures for calculating similarity (e.g., 'jaccard', 'cosine').
        :return: Dictionary of similarity scores with public dataset names as keys.
        """
        scores = {}
        for benchmark_dataset in self.benchmark_datasets:
            score = self.similarity_calculator.calculate_similarity(self.unseen_dataset_name, benchmark_dataset,
                                                                    measures)
            average_score = sum(score.values()) / len(score)  # Averaging the scores
            scores[benchmark_dataset] = average_score
        return scores

    def query_similarity(self, langchain_query, math_similarities=None):
        """
        Queries the LLM for textual similarity between the unseen dataset and public datasets.
        :param langchain_query: Function to query the LLM.
        :param math_similarities: A dictionary with similarity scores.
        :return: The LLM's response as a string.
        """
        # Generate the LLM prompt using the descriptions obtained from the first stage
        if self.use_parser:
            #prompt, user_input, similarity_tool = LLMPromptTemplate.generate_task_similarity_prompt_parser(
            #    self.unseen_dataset_description, self.benchmark_dataset_descriptions, math_similarities)
            prompt, user_input, similarity_tool = LLMPromptTemplate.generate_task_similarity_prompt_parser_metric(
                self.unseen_dataset_description, self.benchmark_dataset_descriptions, math_similarities)
            chain = prompt | langchain_query.with_structured_output(convert_to_openai_function(similarity_tool))
            response = chain.invoke({"input": user_input})
        else:
            prompt = LLMPromptTemplate.generate_task_similarity_prompt_metric(self.unseen_dataset_description,
                                                                              self.benchmark_dataset_descriptions,
                                                                              math_similarities)
            response = langchain_query.invoke(prompt)
            # prompt = LLMPromptTemplate.generate_task_similarity_prompt_llama(self.unseen_dataset_description,
            #                                                                  self.benchmark_dataset_descriptions,
            #                                                                  math_similarities)

            # from transformers import AutoTokenizer
            # cache_dir = r'D:\Artificial_Intelligence\\cache'
            # model_dir = "meta-llama/Llama-2-7b-hf"  # Meta-Llama-3-8B
            # tokenizer = AutoTokenizer.from_pretrained(model_dir,
            #                                           cache_dir=cache_dir,
            #                                           use_fast=True)
            # input_ids = tokenizer(prompt, return_tensors='pt').input_ids
            # prompt_length = input_ids.shape[1]
            # print(f"Prompt length in tokens: {prompt_length}")
            # import torch
            # for dataset_name, description in self.benchmark_dataset_descriptions.items():
            #     prompt = LLMPromptTemplate.generate_task_similarity_prompt_llama(self.unseen_dataset_description,
            #                                                                      description, dataset_name)
            #     with torch.no_grad():
            #         response = langchain_query.invoke(prompt)
            #     print('###')
            #     print(response)
            #     print('###')
            # exit()

        return response

    @staticmethod
    def write_similarity_report(textual_similarity_response, task_similarity_file_name):
        # Open the file in write mode
        with open(task_similarity_file_name, 'w') as file:
            # Iterate over each key that ends with '_similarity_score' to ensure pairs are processed
            for key in sorted(textual_similarity_response):
                if key.endswith('_similarity_score'):
                    dataset_name = key.replace('_similarity_score', '')
                    # Construct the reason key from the dataset name
                    reason_key = f"{dataset_name}_reason"

                    # Retrieve score and reason from the data dictionary
                    score = textual_similarity_response[key]
                    reason = textual_similarity_response.get(reason_key, 'Reason not provided')

                    # Write the formatted output to the file
                    file.write(f"Dataset: {dataset_name}\n")
                    file.write(f"Similarity Score: {score}\n")
                    file.write(f"Reason: {reason}\n")
                    file.write("\n")

    @staticmethod
    def analyze_similarity_scores_from_dict(data, unseen_dataset, top_n=1):
        """
        Analyzes similarity scores from a dictionary to determine the top N similar public datasets to an unseen dataset.

        :param data: Dictionary containing similarity scores and reasons.
        :param unseen_dataset: The name of the unseen dataset being compared.
        :param top_n: The number of top similar datasets to return.
        :return: A tuple containing the list of top N similar public datasets and their similarity scores.
        """
        dataset_scores = []

        # Extract dataset names and scores
        for key, value in data.items():
            if key.endswith('_similarity_score'):
                dataset_name = key.split('_similarity_score')[0]
                score = value
                dataset_scores.append((dataset_name, score))

        # Sort by similarity score in descending order
        sorted_data = sorted(dataset_scores, key=lambda x: x[1], reverse=True)

        # Select the top N datasets
        top_datasets = [dataset[0] for dataset in sorted_data[:top_n]]
        similarity_scores = [dataset[1] for dataset in sorted_data[:top_n]]

        # Format the output similar to the previous function's output
        formatted_top_datasets = {unseen_dataset: top_datasets}
        formatted_similarity_scores = {
            unseen_dataset: {name: score for name, score in zip(top_datasets, similarity_scores)}}

        return formatted_top_datasets, formatted_similarity_scores

    @staticmethod
    def analyze_similarity_scores_from_log(similarity_response_file_path, unseen_dataset, top_n=1):
        """
        Analyzes similarity scores to determine the top N similar public datasets to an unseen dataset.

        :param similarity_response_file_path: Path to the file containing similarity scores.
        :param unseen_dataset: The name of the unseen dataset being compared.
        :param top_n: The number of top similar datasets to return.
        :return: A tuple containing the list of top N similar public datasets and their similarity scores.
        """
        data = []
        similarity_line = False
        with open(similarity_response_file_path, 'r') as file:
            for line in file:
                '''
                if line.startswith('-'):
                    parts = line.split(':')
                    datasets, score = parts[0], parts[1]
                    _, d2 = datasets.split('and')
                    d2 = d2.strip()
                    score = float(score.strip())
                    data.append([d2, score])
                '''
                if "Unseen Dataset" in line and ':' in line:
                    # Extract the parts of the line containing dataset names and the score
                    parts = line.split(':')
                    dataset_info, score = parts[0].strip(), parts[1].strip()
                    _, dataset_name = dataset_info.split("and")
                    dataset_name = dataset_name.replace("*", "").strip()
                    try:
                        score = float(score.replace("*", "").strip())
                        data.append([dataset_name, score])
                    except ValueError:
                        similarity_line = True
                elif similarity_line:
                    parts = line.split(':')
                    _, score = parts[0].strip(), parts[1].strip()
                    score = float(score.replace("*", "").strip())
                    data.append([dataset_name, score])
                    similarity_line = False

        # Sort by similarity score in descending order and select the top N
        filtered_data = sorted(data, key=lambda x: x[1], reverse=True)[:top_n]

        # Construct the return format
        top_datasets = {unseen_dataset: [item[0] for item in filtered_data]}
        similarity_scores = {unseen_dataset: {item[0]: item[1] for item in filtered_data}}

        return top_datasets, similarity_scores


def analyze_self_evaluation_results(similarity_response_file_path, top_n=1):
    """
    Analyzes both mathematical and textual similarities to determine the top 3 similar datasets.
    :param similarity_response_file_path: String response from the LLM regarding textual similarities.
    :return: List of top n similar public datasets based on the analysis.
    """
    # Assuming the response is a list-like string: "1. DatasetA, 2. DatasetB, 3. DatasetC, ..."
    #response_lines = textual_similarity_response.split(", ")
    #top_datasets = [line.split(". ")[1] for line in response_lines[:3]]  # Extract the first 3 dataset names
    data = []
    with open(similarity_response_file_path, 'r') as file:
        for line in file:
            if line.startswith('-'):
                parts = line.split(':')
                datasets, score = parts[0], parts[1]
                d1, d2 = datasets.split('and')
                d1, d2 = d1.strip().replace('-', '').strip(), d2.strip()
                score = float(score.strip())
                data.append([d1, d2, score])

    df = pd.DataFrame(data, columns=['Dataset1', 'Dataset2', 'Similarity'])
    unique_datasets = ["Cora", "CiteSeer", "PubMed", "CS", "Physics", "Photo", "Computers", "ogbn-arxiv"]
    matrix = pd.DataFrame(np.nan, index=unique_datasets, columns=unique_datasets)

    # Populate the matrix
    for i, row in df.iterrows():
        matrix.loc[row['Dataset1'], row['Dataset2']] = row['Similarity']
        matrix.loc[row['Dataset2'], row['Dataset1']] = row['Similarity']

    # Fill diagonal with 1.0 for self-similarity
    np.fill_diagonal(matrix.values, 1.0)
    most_similar, similarity_scores = find_most_similar_datasets_variable_with_scores(matrix, top_n)
    return most_similar, similarity_scores


def find_most_similar_datasets_variable_with_scores(df, top_n=1):
    most_similar_variable = {}
    similarity_scores = {}
    for index, row in df.iterrows():
        # Exclude the similarity to itself by setting it to -1
        filtered_row = row.mask(row == 1, -1)
        # Sort the similarities in descending order
        sorted_similarities = filtered_row.sort_values(ascending=False)
        # Select the top N similar datasets, taking care of possible ties
        if top_n > 1:
            # Ensure we include all datasets that are tied at the Nth position
            threshold_similarity = sorted_similarities.iloc[top_n - 1]
            top_similar_datasets_indices = sorted_similarities[sorted_similarities >= threshold_similarity].index
        else:
            # For top_n = 1, simply select the dataset(s) with the highest similarity, including ties
            max_similarity = sorted_similarities.max()
            top_similar_datasets_indices = sorted_similarities[sorted_similarities == max_similarity].index

        # Update most_similar_variable with the top similar datasets
        most_similar_variable[index.lower()] = top_similar_datasets_indices.str.lower().tolist()

        # Update similarity_scores with the similarity scores for the top similar datasets
        similarity_scores[index.lower()] = {dataset.lower(): row[dataset] for dataset in top_similar_datasets_indices}

    return most_similar_variable, similarity_scores

