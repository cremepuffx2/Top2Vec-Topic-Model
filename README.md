# Top2Vec-Topic-Model
<p align="center">
  <img src="https://github.com/user-attachments/assets/c5745603-e221-4709-8963-83c4ccaeb7f5" width="500"/>
</p>

## Coursera Topic Modeling with Top2Vec

This repository contains a notebook that applies the Top2Vec algorithm to perform topic modelling on Coursera programme module summaries. The goal is to automatically discover coherent topics from a large collection of course descriptions, enabling curriculum designers and learners to quickly understand the thematic structure of hundreds of programmes. 

Top2Vec is a state‑of‑the‑art model for topic modeling and semantic search. It automatically detects topics present in text and jointly embeds topics,
documents and words in the same vector space. Unlike traditional approaches such as Latent Dirichlet Allocation (LDA), Top2Vec requires no pre‑defined number of topics, no stop‑word lists and no stemming/lemmatization. These properties make it well‑suited for noisy or short text like course summaries. The algorithm works by creating jointly embedded document and word vectors, reducing the document vectors into a lower dimensional space, clustering them to find dense areas, computing topic centroids and then extracting the nearest words as topic keywords.

## Problem Statement
Large learning platforms host thousands of courses and specialisations. Forcurriculum designers and students it becomes increasingly difficult to identify overlapping themes or gaps among programmes. This project uses Top2Vec to uncover latent topics from Coursera programme module outlines. By clustering modules into meaningful themes, we hope to provide insights into how different programmes relate to each other and to guide future programme development.

## Dataset
The dataset consists of 475 programme files scraped from Coursera. Each file (program0.txt through program474.txt) contains multiple module summaries wrapped between <DOC> and </DOC> tags. The notebook uses a helper function load_program_documents to extract the individual documents from each file. Collectively, these files contain thousands of short descriptions covering a wide range of topics, from computer science and data science to personal development.

## Data Loading
First, we define a function to extract all documents from one programme file, then we aggregate all the douments obtained across all 475 programmes

## Top2Vec Overview
Top2Vec provides several advantages over classical topic modelling:

Automatic topic number – the model determines how many topics exist in the data, so there is no need to choose k in advance.
Minimal preprocessing – no stop‑word lists, stemming or lemmatisation are required.
Short text support – works effectively with short documents such as course summaries.
Joint embeddings – generates embeddings for documents, words and topics in the same space, enabling similarity search among them.

The algorithm consists of five key steps:

Joint Embedding: Create jointly embedded document and word vectors using Doc2Vec, Universal Sentence Encoder or a BERT sentence transformer.
Dimensionality Reduction: Reduce the high‑dimensional document vectors into a lower dimensional space using UMAP to make density estimation feasible.
Clustering: Apply HDBSCAN to the reduced vectors to find dense clusters; outliers are ignored.
Topic Centroids: For each dense cluster, compute the centroid of the original high‑dimensional document vectors to obtain the topic vector.
Keyword Extraction: For each topic vector, retrieve the n closest word vectors to obtain the most representative keywords.

The notebook uses the default Doc2Vec model for embedding and passes a minimum cluster size of 40 to HDBSCAN, which controls how many documents must form a dense region to be considered a topic. Setting a larger minimum cluster size encourages fewer, broader topics.

### Pre‑processing and Training Pipeline
The preprocessing steps involve the following:
- Import libraries and dataset – the notebook loads required libraries such as pandas, numpy, top2vec, umap-learn and gensim, then extracts all documents from the Coursera corpus.
- Train Top2Vec model – instantiate a Top2Vec model on the list of documents using the default Doc2Vec embedding. We set hdbscan_args={'min_cluster_size': 40} and use workers=15 to speed up training:
- Save the trained model
- Inspect topics – after training, we use model.get_topics() to return the topic words, their scores and the topic indices. We print the top 10 words for each topic and compute the number of topics detected.
<p align="center">
  <img src="https://github.com/user-attachments/assets/6ec7aab7-5ae8-4abe-9a30-76b193dee73f" width="550"/>
</p>
<p align="center"><i>Figure 1: C_v Coherence Score Formula for Topic Modelling</i></p>
- Coherence scoring – to evaluate topic quality, we compute the c_v coherence score using gensim’s CoherenceModel. We tokenise the documents, build a dictionary and pass the top words per topic. A higher coherence score indicates more semantically consistent topics.



## Visualisation and Exploration
<img width="1003" height="537" alt="image" src="https://github.com/user-attachments/assets/31d87638-e02f-4e3b-b7b4-8ccc3c9ff7cf" />
<p align="center"><i>Figure 2: Word Cloud of Topic 0 generated using Top2Vec</i></p>

To gain insight into the learned topics, the notebook projects the high‑dimensional topic vectors into two dimensions using UMAP. Each topic is labelled with its top three keywords and plotted in a scatter plot. Clusters that are closer in 2D space represent semantically similar themes. We also generate word clouds for individual topics to visually explore the distribution of keywords.

## Results and Discussion
After training on the Coursera module summaries, Top2Vec discovered multiple coherent themes. For example, some topics were dominated by keywords related to data science (e.g., “data”, “analysis”, “machine”), while others captured business or social sciences. The UMAP plot revealed that similar topics clustered closely together, suggesting that the algorithm effectively grouped related programmes.

The coherence score computed via gensim provided a quantitative measure of topic quality. A preliminary run yielded a c_v score of around 0.45, indicating moderate semantic consistency across topics. Increasing the minimum cluster size or experimenting with alternative embedding models (e.g., Universal Sentence Encoder) could further improve coherence.

## Future Work
Some of the future work that may be undertaken to build upon the insights derived from this project could include:
- Embedding variations: Explore pre‑trained universal sentence encoders or BERT-based sentence transformers by passing embedding_model='universal-sentence-encoder' or embedding_model='distiluse-base-multilingual-cased' to the Top2Vec constructor.
- Fine‑tuning HDBSCAN: Adjust the min_cluster_size and other HDBSCAN parameters to balance topic granularity and noise suppression.
- Integration into recommendation systems: Use the topic embeddings to build a semantic search engine for course recommendations or to identify gaps in current programme offerings.
- Hierarchical topics: Top2Vec supports hierarchical topics which could offer a coarse‑to‑fine view of themes
