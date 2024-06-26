
# Clustering Patent Claim Sentences
This project focuses on extracting, processing, and clustering patent claim sentences from specified URLs. Various Natural Language Processing (NLP) techniques and clustering algorithms are used to group similar sentences and assign topics to them. Additionally, the project features a Flask web application where users can choose the number of clusters for the clustering algorithm. Once the clustering is performed, users can view the output, which includes the title for each group and the number of sentences in each group.

## Clustering Methodology

This project leverages Word Embeddings with Word2Vec and the KMeans clustering model to cluster patent claim sentences, offering several benefits:

1. **Semantic Understanding**: Word Embeddings with Word2Vec enable the model to understand the meaning and context of words, allowing for more accurate clustering based on semantic similarity rather than just keyword matching.

2. **Efficient Grouping**: KMeans clustering efficiently groups patent claim sentences by iteratively assigning them to clusters based on the similarity of their word embeddings. This results in clusters that contain sentences with similar meanings or topics.

3. **Scalability**: Both Word2Vec and KMeans are scalable techniques that can handle large amounts of data, making them suitable for processing and clustering patent claim sentences from a variety of sources.

4. **Interpretability**: The clusters generated by this methodology are interpretable, as each cluster represents a group of sentences that share similar semantic content or topics. This allows users to quickly grasp the main themes present in the patent documents.

Overall, the combination of Word Embeddings with Word2Vec and KMeans clustering provides an efficient and effective way to group patent claim sentences, facilitating the analysis and exploration of patent documents.


## How to run the project
##### This project used Python 3.11.3
In order to run this project there is two diffrenet options:

1. 
  - Push on `code` and download the zip.
  - Extract the zip package.
  - Open the package in Python environment.
  - Open terminal and run python3.11 -m venv myenv
    - On Windows - run myenv\Scripts\activate
    - On macOS and Linux - source myenv/bin/activate
  - Run pip install -r requirements.txt
  - Run python3 app.py -> for the web application
    - Open Google and insert the provided URL
    - Insert number of cluster to see result 
  - Run the Task_2.ipynb
    - To see the three methods you can manually adjust the number of clusters in the second section.
Now you can successfully use the project.

2.
  - Push on `code` and download the zip.
  - Extract the zip package.
  - Open the package in Python environment.
  - Open Docker Desktop
  - Open terminal and run 
    - docker build -t machine .
    - docker run -p 5005:5005 machine 
  - After this you will successfully run the application
    - Open Google and insert the provided URL 
    - Insert number of cluster to see result
  
#### Note
Additionally, another option is available: by adding an API key and changing the flag in the `app_model.py` file to true, you can utilize a method to automatically generate titles based on the conversationwith chatGPT.

