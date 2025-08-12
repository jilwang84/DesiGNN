# DesiGNN
 DesiGNN: Proficient Graph Neural Network Design by Accumulating Knowledge on Large Language Models
 
WSDM 2026 Submission Number: 558

We present our supplementary material in the file: Supplementary Material.pdf. This includes our complete experiment settings, detailed algorithm, full results, and comprehensive studies. Furthermore, to study the generalization capability of the DesiGNN pipeline to other data and tasks, we present the following study. 

Specifically, there is no significant barrier to extending our DesiGNN to graph classification or link prediction, except for replacing the benchmark knowledge base and mode design space. Our methodology is itself a generalizable approach and can utilize most existing benchmark sources. To support, we replace the NAS-Bench-Graph benchmark knowledge with an SOTA design space [1] that is able to tackle link prediction and graph classification on more diverse graph structures. As shown in the tables below, DesiGNN surpasses the baselines in all tasks, demonstrating strong generalization and plug-and-play utility. KBG [2] is a new SOTA baseline. 

**Node Classification:**
| **Method** | **Actor** | **Computers** | **Photo** | **CiteSeer** | **CS** | **Cora** | **Cornell** | **DBLP** | **PubMed** | **Texas** | **Wisconsin** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Random | 33.98 | 88.25 | 94.28 | 73.84 | 94.96 | 87.84 | 74.96 | 83.44 | 88.46 | 80.63 | 89.93 |
| RL | 33.95 | 88.25 | 94.37 | 73.88 | 95.02 | 88.01 | 74.87 | 83.40 | 88.58 | 81.62 | 89.73 |
| EA | 33.73 | 88.28 | 94.45 | 73.93 | 94.94 | 87.90 | 74.24 | 83.66 | 88.42 | 82.43 | 89.20 |
| GNAS | 34.11 | 87.94 | 94.38 | 73.89 | 94.90 | 88.01 | 74.60 | 83.40 | 88.58 | 81.68 | 89.87 |
| Auto-GNN | 33.71 | 87.59 | 94.46 | 74.14 | 95.15 | 87.94 | 74.51 | 83.69 | 88.38 | 81.71 | 89.60 |
| Kendall | 32.72 | 82.57 | 93.53 | 72.68 | 94.71 | 87.95 | 69.37 | 82.54 | 87.61 | 74.77 | 83.33 |
| Overlap | 32.17 | 82.57 | 93.70 | 65.67 | 93.88 | 88.07 | 69.37 | 83.47 | 88.66 | 74.77 | 86.67 |
| KBG | 32.72 | 87.38 | 94.53 | 74.23 | 94.71 | 87.95 | 69.37 | 83.47 | 88.15 | 74.77 | 86.67 |
| AutoTransfer | 33.97 | 87.72 | **94.62** | 73.89 | **95.16** | **88.50** | **75.58** | 83.59 | **89.08** | 78.38 | 88.67 |
| **DesiGNN** | **34.43** | **88.40** | **94.60** | **74.54** | 95.03 | 88.34 | **75.50** | **84.29** | **89.08** | **81.80** | **90.66** |

**Link Prediction:**
| **Method** | **Actor** | **Computers** | **Photo** | **CiteSeer** | **CS** | **Cora** | **Cornell** | **DBLP** | **PubMed** | **Texas** | **Wisconsin** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Random | 73.76 | 93.26 | 93.94 | 74.24 | 89.38 | 74.66 | 73.84 | 85.48 | 83.13 | 68.70 | 68.98 |
| RL | 73.92 | 93.19 | 94.06 | 74.24 | 89.26 | **76.40** | 73.38 | 85.37 | 83.60 | 69.07 | 68.30 |
| EA | 73.88 | 92.89 | 94.17 | 74.00 | 89.30 | 75.22 | 74.60 | 85.54 | 83.76 | 68.52 | 68.68 |
| GNAS | 73.97 | 93.04 | 93.99 | 74.18 | 89.48 | 76.30 | 74.40 | 85.75 | 83.87 | 69.95 | 68.47 |
| Auto-GNN | 74.58 | 93.34 | 94.08 | 74.68 | 89.46 | 74.09 | 75.36 | 85.10 | 82.86 | 70.18 | 69.93 |
| Kendall | 72.35 | 86.70 | 93.00 | 71.91 | 87.63 | 70.88 | 69.19 | 85.00 | 74.88 | 66.67 | 66.33 |
| Overlap | 70.33 | 92.94 | 93.00 | 71.91 | 87.63 | 70.88 | 64.14 | 84.46 | 82.54 | 66.67 | 66.67 |
| AutoTransfer | 74.65 | **93.61** | 93.98 | 73.35 | 90.13 | 75.83 | 74.75 | 85.75 | 84.86 | **72.15** | 70.07 |
| **DesiGNN** | **74.75** | **93.61** | **94.33** | **75.11** | **90.26** | 75.67 | **76.77** | **86.56** | **85.55** | 71.29 | **71.77** |

**Graph Classification:**
| **Method** | **COX2** | **DD** | **IMDB-BINARY** | **IMDB-MULTI** | **NCI1** | **NCI109** | **PROTEINS** | **PTC_FM** | **PTC_FR** | **PTC_MM** | **PTC_MR** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Random | 85.24 | 79.86 | 79.98 | 47.33 | 78.82 | 76.47 | 78.53 | 69.04 | 66.71 | 69.90 | 62.11 |
| RL | 84.80 | 79.22 | 80.08 | 47.35 | 78.27 | 76.71 | 78.92 | 69.18 | 66.10 | **70.55** | 61.52 |
| EA | 84.77 | 79.16 | 79.28 | 47.29 | 78.33 | 76.49 | 78.67 | 68.75 | 65.76 | 69.06 | 61.13 |
| GNAS | 85.20 | 79.57 | 80.00 | 47.32 | 78.26 | 76.75 | 78.83 | 68.70 | 66.47 | 70.20 | 61.42 |
| Auto-GNN | 85.41 | 79.69 | 79.12 | 46.97 | 77.73 | 76.24 | 77.69 | 68.12 | 65.05 | 70.50 | 59.36 |
| Kendall | 72.76 | 73.90 | 72.17 | 46.56 | 74.49 | 69.25 | 76.13 | 65.22 | 59.05 | 59.20 | 50.49 |
| Overlap | 82.44 | 62.84 | 75.83 | 44.67 | 74.49 | 69.25 | 77.63 | 47.34 | 63.34 | 58.70 | 54.90 |
| AutoTransfer | 85.74 | 78.44 | 80.17 | 47.22 | **79.00** | **77.74** | 78.38 | 67.15 | **67.14** | 70.15 | 61.76 |
| **DesiGNN** | **85.95** | **80.04** | **80.73** | **47.44** | **78.95** | 77.58 | **79.73** | **69.49** | **67.14** | **70.55** | **62.25** |

**Reference:**

[1] Wang, Z., Di, S., & Chen, L. (2023, August). A message passing neural network space for better capturing data-dependent receptive fields. In Proceedings of the 29th ACM SIGKDD conference on knowledge discovery and data mining (pp. 2489-2501).

[2] Liu, H., Di, S., Wang, J., Wang, Z., Wang, J., Zhou, X., & Chen, L. Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Designing Models. In The Thirteenth International Conference on Learning Representations, ICLR 2025.