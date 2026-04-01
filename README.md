## Top2Vec-Topic-Model
# Coursera Topic Modeling with Top2Vec

This repository contains a notebook that applies the Top2Vec algorithm to perform topic modelling on Coursera programme module summaries. The goal is to automatically discover coherent topics from a large collection of course descriptions, enabling curriculum designers and learners to quickly understand the thematic structure of hundreds of programmes. 

Top2Vec is a state‑of‑the‑art model for topic modeling and semantic search. It automatically detects topics present in text and jointly embeds topics,
documents and words in the same vector space. Unlike traditional approaches such as Latent Dirichlet Allocation (LDA), Top2Vec requires no pre‑defined number of topics, no stop‑word lists and no stemming/lemmatization. These properties make it well‑suited for noisy or short text like course summaries. The algorithm works by creating jointly embedded document and word vectors, reducing the document vectors into a lower dimensional space, clustering them to find dense areas, computing topic centroids and then extracting the nearest words as topic keywords.

# Problem Statement
Large learning platforms host thousands of courses and specialisations. Forcurriculum designers and students it becomes increasingly difficult to identify overlapping themes or gaps among programmes. This project uses Top2Vec to uncover latent topics from Coursera programme module outlines. By clustering modules into meaningful themes, we hope to provide insights into how different programmes relate to each other and to guide future programme development.

# Dataset
The dataset consists of 475 programme files scraped from Coursera. Each file (program0.txt through program474.txt) contains multiple module summaries wrapped between <DOC> and </DOC> tags. The notebook uses a helper function load_program_documents to extract the individual documents from each file. Collectively, these files contain thousands of short descriptions covering a wide range of topics, from computer science and data science to personal development.

# Data Loading
# Function to extract documents from one programme file
def load_program_documents(program_index: int) -> List[str]:
    file_path = f"../corpus/program{program_index}.txt"
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # Extract content inside <DOC>...</DOC>
    docs = re.findall(r'<DOC>(.*?)</DOC>', content, re.DOTALL)
    return docs

# Aggregate all documents across 475 programmes
all_documents = []
for i in range(475):
    docs = load_program_documents(i)
    if docs:
        all_documents.extend(docs)
print(f"Total documents loaded: {len(all_documents)}")

# Top2Vec Overview
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

# Pre‑processing and Training Pipeline
Import libraries and dataset – the notebook loads required libraries such as pandas, numpy, top2vec, umap-learn and gensim, then extracts all documents from the Coursera corpus.

Train Top2Vec model – instantiate a Top2Vec model on the list of documents using the default Doc2Vec embedding. We set hdbscan_args={'min_cluster_size': 40} and use workers=15 to speed up training:

from top2vec import Top2Vec
model = Top2Vec(documents=all_documents,
                 hdbscan_args={'min_cluster_size': 40},
                 workers=15)

# Save the trained model
model.save('top2vec_module')
Inspect topics – after training, we use model.get_topics() to return the topic words, their scores and the topic indices. We print the top 10 words for each topic and compute the number of topics detected.
Coherence scoring – to evaluate topic quality, we compute the c_v coherence score using gensim’s CoherenceModel. We tokenise the documents, build a dictionary and pass the top words per topic. A higher coherence score indicates more semantically consistent topics.
Save and load – Top2Vec models can be serialised and loaded later using model.save() and Top2Vec.load().

# Visualisation and Exploration
To gain insight into the learned topics, the notebook projects the high‑dimensional topic vectors into two dimensions using UMAP. Each topic is labelled with its top three keywords and plotted in a scatter plot. Clusters that are closer in 2D space represent semantically similar themes. We also generate word clouds for individual topics to visually explore the distribution of keywords.

Below is a conceptual diagram illustrating the Top2Vec workflow used in this project:

Reproducing the Results

To reproduce this project on your own machine:

Install dependencies – Top2Vec can be installed via pip:

pip install top2vec
pip install top2vec[sentence_encoders]  # optional for universal sentence encoder
pip install top2vec[sentence_transformers]  # optional for BERT-based embeddings
Prepare the dataset – create a corpus/ folder in the notebook’s parent directory and add your scraped Coursera programme files named program0.txt, program1.txt, … up to program474.txt. Ensure each module description is wrapped between <DOC> and </DOC> tags.
Run the notebook – execute the cells in order. Training may take some time depending on the number of documents and your hardware. Adjust min_cluster_size and other parameters to fine‑tune topic granularity.
Explore topics – use model.get_num_topics(), model.get_topic_sizes() and model.get_topics() to inspect the discovered themes. You can also perform semantic search or query the most similar documents to a given topic using built‑in functions.

# Results and Discussion
After training on the Coursera module summaries, Top2Vec discovered multiple coherent themes. For example, some topics were dominated by keywords related to data science (e.g., “data”, “analysis”, “machine”), while others captured business or social sciences. The UMAP plot revealed that similar topics clustered closely together, suggesting that the algorithm effectively grouped related programmes.

The coherence score computed via gensim provided a quantitative measure of topic quality. A preliminary run yielded a c_v score of around 0.45 (your value may vary depending on the dataset and parameters), indicating moderate semantic consistency across topics. Increasing the minimum cluster size or experimenting with alternative embedding models (e.g., Universal Sentence Encoder) could further improve coherence.

# Future Work
Embedding variations: Explore pre‑trained universal sentence encoders or BERT-based sentence transformers by passing embedding_model='universal-sentence-encoder' or embedding_model='distiluse-base-multilingual-cased' to the Top2Vec constructor.
Fine‑tuning HDBSCAN: Adjust the min_cluster_size and other HDBSCAN parameters to balance topic granularity and noise suppression.
Integration into recommendation systems: Use the topic embeddings to build a semantic search engine for course recommendations or to identify gaps in current programme offerings.
Hierarchical topics: Top2Vec supports hierarchical topics which could offer a coarse‑to‑fine view of themes; this has yet to be explored in the current notebook.
