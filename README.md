**Hybrid CNN-GNN Architecture for Infrastructure Classification in Satellite Imagery**

This repository contains the implementation of a hybrid deep learning model that combines Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs)for infrastructure classification in satellite imagery. The project specifically focuses on classifying Residential, Industrial, and Highway categories within the EuroSAT Sentinel-2 dataset.

**The Core Problem**
Traditional CNNs are excellent at learning local visual patterns like textures and shapes. However, they often fail to capture the larger spatial context—how distant but related objects like roads and buildings are positioned relative to one another. This research addresses these limitations by using GNNs to model the irregular, non-Euclidean spatial relationships common in remote sensing data.

**Key Results**
  - Classification Accuracy: 98.1% achieved by the hybrid model.
  - F1-Score: 0.979, demonstrating robust performance across imbalanced classes.
  - Methodological Rigor: Evaluation performed using a geographically held-out split to ensure the model generalizes to entirely unseen geographic regions

**Methodology & Architecture**
The pipeline follows a sophisticated multi-stage approach to transform raw pixels into relational knowledge:
  - Feature Extraction (CNN): Each image passes through a ResNet18 backbone to extract high-level spatial features, capturing building geometries and local land surface textures.
  - Graph Construction: Feature maps are converted into graphs where nodes represent image patches (superpixels) and edges represent spatial adjacency or visual similarity.
  - Relational Reasoning (GNN): Two message-passing layers enable multi-hop reasoning, allowing the model to learn contextual dependencies between nearby and distant regions.
  - Classification: A Multi-Layer Perceptron (MLP) processes the final graph-level representation to predict the infrastructure category.

**Research & Findings**
  - The hybrid approach provides a "Spatial Reasoning" advantage that standalone architectures lack:
  - Contextual Awareness: While a CNN might see a "roof," the GNN understands its relationship to surrounding infrastructure, reducing "misclassification noise" at boundaries between industrial and residential zones.
  - Efficiency: GNNs explore topological relationships with fewer layers and lower computational power compared to deep CNN stacks.
  - Interpretability: The graph structure provides a more transparent look at how the model relates different geographic features to reach a classification decision.

**Tech Stack**
  - Programming: Python.
  - Deep Learning: PyTorch, Torch-Geometric (for GNN implementation).
  - Computer Vision: SLIC (Simple Linear Iterative Clustering for segmentation).
  - Deployment: Flask & Hugging Face.

**Future Work**
  - Based on the thesis conclusions, future iterations of this work include:
  - Multi-class Expansion: Moving beyond binary classification to include more diverse land cover types.
  - Multi-spectral Integration: Incorporating additional Sentinel-2 bands beyond RGB to increase performance.
  - Cross-region Generalization: Testing on satellite data from entirely different continents to assess global robustness.

Developed as a Master's Thesis for the MSc in Data Science & Analytics at the University of Westminster (2025).
