# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license
# This file combines the graph description with user input or predefined descriptions.


class DescriptionCombiner:
    def __init__(self, dataset_name, user_description, is_benchmark, num_hops, translations=None):
        """
        Initialize the DescriptionCombiner with a user description, the name of the dataset, graph metrics (translations),
        a flag indicating if the dataset is a benchmark, and the number of hops for subgraph sampling.

        :param dataset_name: The name of the dataset.
        :param user_description: A textual description provided by the user about the dataset.
        :param translations: A dictionary containing various graph metrics.
        :param is_benchmark: A boolean indicating if the dataset is a benchmark dataset.
        :param num_hops: An integer indicating the number of hops for subgraph sampling.
        """
        self.dataset_name = dataset_name
        self.user_description = user_description
        self.translations = translations
        self.is_benchmark = is_benchmark
        self.num_hops = num_hops

    def format_translations(self):
        """
        Formats the translations (graph metrics) into categories for better readability.

        :return: A string representation of the graph metrics categorized into local and general metrics.
        """
        # Splitting the dictionary for clearer categorization
        local_metrics = {key: self.translations[key] for key in self.translations if 'local_' in key}
        general_metrics = {key: self.translations[key] for key in self.translations if 'local_' not in key}

        formatted_str = f"{self.num_hops}-Hop Subgraph Metrics:\n"
        for key, value in local_metrics.items():
            # Removing 'local_' from the key for printing
            formatted_key = key.replace('local_', '').replace('_', ' ').title()
            formatted_str += f"- {formatted_key}: {value}\n"

        # Adding a space between the sections for better readability
        formatted_str += "\nGeneral Graph Metrics:\n"
        for key, value in general_metrics.items():
            # Formatting the key to make it more readable
            formatted_key = key.replace('_', ' ').title()
            formatted_str += f"- {formatted_key}: {value}\n"

        return formatted_str

    def combine_descriptions(self):
        """
        Combines the user description, dataset name, graph metrics, dataset type (benchmark or unseen),
        and k-hop subgraphs into a single text, and formulates a question for the LLM.

        :return: A combined description and question as a string.
        """
        dataset_type = "Benchmark" if self.is_benchmark else "Unseen"
        if dataset_type == "Benchmark":
            combined_description = f"Dataset Name: {self.dataset_name}\n"
        else:
            combined_description = ""
        combined_description += f"Dataset Type: {dataset_type}\n"
        combined_description += f"User Description:\n{self.user_description}\n\n"

        # Add formatted translations (graph metrics)
        if self.translations:
            combined_description += self.format_translations()

        return combined_description

