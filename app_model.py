from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import requests
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
import re
from chat import Chat

FLAG = False

class Model:
    def __init__(self , urls):
        self.__urls__ = urls
        self.__test__ = None
        self.__data__ = None
        
    def print_the_results(self, labels, sentences):
        """
        Prints the results of clustering sentences.

        :param labels: List of cluster labels assigned to each sentence
        :param sentences: List of sentences to be clustered
        """
        # Initialize a dictionary to store sentences for each cluster
        cluster_dict = {}

        # Filling up the cluster dictionary with data. 
        for cluster_num, sentence in zip(labels, sentences):
            if cluster_num not in cluster_dict:
                cluster_dict[cluster_num] = [sentence]
            else:
                cluster_dict[cluster_num].append(sentence)

        # Initialize lists to store aggregated text and number of claims per cluster
        all_text = []
        array_num_of_claims = []
        text = ""

        # Aggregate text for each cluster and count the number of claims
        for cluster_num, cluster_sentences in cluster_dict.items():
            number_of_claims = 0
            for sentence in cluster_sentences:
                number_of_claims += 1
                # Group all the sentences from the same cluster
                text += " " + sentence
            all_text.append(text)
            array_num_of_claims.append(number_of_claims)
            text = ""

        # Get titles for each aggregated text
        titles , chat_title = self.topic_render(all_text)

        agg_title = {}

        # Update the aggregated titles with their numbers of claims from the array_num_of_claims extracting by index
        for title in titles:
            # Check if the title is already in the aggregated titles dictionary
            if title[0] in agg_title:
                # If yes, add the number of claims to the existing count
                agg_title[title[0]] += array_num_of_claims[title[1]-1]
            else:
                # If no, initialize the count with the number of claims
                agg_title[title[0]] = array_num_of_claims[title[1]-1]
        data = []
        # Print the aggregated titles and their numbers of claims
        for title, claims in agg_title.items():
            data.append(f"title: {title}, numbers of claims: {claims}")

        if FLAG:
            data.append(f"---> Also the chat provide titles:")
            # Convert new data tuples to the desired format and append them to the existing list
            for title, count in chat_title:
                data.append(f"title: {title}, numbers of claims: {array_num_of_claims[count]}")
        self.__data__ = data

    def url_reader(self, url):
        """
        Scrapes the webpage and extracts sentences containing specified keywords.
        """
        # Keyword to search to start from this line
        keyword = 'Claims'

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title and type of patent from webpage title
            topic_title = soup.title.string
            components = topic_title.split(" - ")
            # Patent title
            title = components[1]
            # Webpage title
            type_patent = components[2]

            # Exclude content within table elements
            for table in soup.find_all('table'):
                table.extract()

            # Find all text on the webpage
            all_text = soup.get_text()

            # Split the text into sentences with periods as separators
            sentences = re.split(r'(?<=[.!?])\s+', all_text)

            # Initialize a flag to indicate when to start adding sentences
            add_sentences = False
            assertion_words = ["claim", "assert", "argue", "propose", "conclude",
                               "suggest", "state", "believe", "affirm", "maintain", "contend", "insist", "apparatus"]

            # List to store sentences containing the keyword
            keyword_sentences = []

            # Iterate through the sentences
            for sentence in sentences:
                # Check if the keyword to start from is found in the sentence
                if keyword in sentence:
                    # Set the flag to True to start adding sentences
                    add_sentences = True

                # Add the sentence if the flag is True and it meets criteria
                if add_sentences and any(word.lower() in sentence.lower() for word in assertion_words) and \
                        sentence.strip().endswith(".") and title not in sentence and type_patent not in sentence \
                        and "Fig" not in sentence and "Claims (" not in sentence:
                    keyword_sentences.append(sentence.strip())

            return keyword_sentences
        else:
            print("Failed to fetch the webpage. Status code:", response.status_code)

    def get_text_test(self):
        claim_sentences = []

        # Data Collection and Preprocessing
        for url in self.__urls__:
            # Fetch and process the text from the URL
            text = self.url_reader(url)
            # Aggregate all the data together
            claim_sentences.extend(text)
        self.__test__ = claim_sentences
    
    
   # Convert each sentence to a vector representation by averaging word vectors
    def sentence_to_vector(self,sentence, model):
        """
        Convert each sentence to a vector representation by averaging word vectors.

        :param sentence: Input sentence to convert
        :param model: Word2Vec model used for word embeddings
        :return: Vector representation of the sentence
        """
        # Tokenize the sentence and filter out words not present in the Word2Vec model
        words = [word for word in sentence.split() if word in model.wv]
        
        # If no words are found in the model, return a zero vector
        if not words:
            return np.zeros(model.vector_size)

        # Get word vectors for the words present in the model
        vectors = [model.wv[word] for word in words]
        
        # Calculate the mean of word vectors to obtain the sentence vector
        return np.mean(vectors, axis=0)


    
    def model_KMeansW2V(self, num_clusters):
        self.get_text_test()
        test_claim_sentences = self.__test__
        # Word2Vec model
        word2vec_model = Word2Vec(
            sentences=[sentence.split() for sentence in test_claim_sentences],  # Convert sentences to word lists
            vector_size=100,  # Dimensionality of word vectors
            window=5,         # Maximum distance between the current and predicted word within a sentence
            min_count=1,      # Ignore words with a frequency lower than this
            workers=4         # Number of threads to train the model
        )

        # Generate Word2Vec vectors for test claim sentences
        test_vectors = np.array([self.sentence_to_vector(sentence, word2vec_model) for sentence in test_claim_sentences])

        # Cluster Modeling (K-means clustering)
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
        
        # Fit KMeans model on the Word2Vec vectors
        kmeans_model.fit(test_vectors)

        # Predict clusters for the test claim sentences
        if len(test_vectors) > 0:
            new_data_clusters = kmeans_model.predict(test_vectors)

            # Print the results of clustering
            self.print_the_results(new_data_clusters, test_claim_sentences)

    def get_result(self):
        return self.__data__
    
    def topic_render(self,text):
        # Define keywords for each title
        topic_keywords = {
        "Wireless": ["wireless", "loss"],
        "Telephone": ["telephone", "phone"],
        "Call": ["call", "indication" , "conversation" , "second party"],
        "Functionality": ["functionality","Efficient" , "switching" , "quality of service"],
        "Communication": ["communication", "source" , "network" , "signal"],
        "Device": ["device" ,"tool","machine"],
        "Service": ["service"],
        "Protocols": ["protocols"],
        "Microphones": ["microphones", "speaker" ,"sound Sensors" ,"acoustic Sensors" , "sound output"],
    }

        # Assign topics to sentences
        titles , chat_titles = self.assign_topic(text , topic_keywords)
        
        return titles , chat_titles


    # Function to assign topic to a sentence based on occurrence of keywords
    def assign_topic(self,text, topic_keywords):
        assigned = []
        # Copy the original text for reuse the text
        remaining_text = text.copy()  
        # Iterate over each topic and evaluate the best score he have for the specific sentences
        for topic, keywords in topic_keywords.items():
            # If there is no more sentences in the text
            if not remaining_text:
                break
            best_topic = "Other"
            best_score = 0 # Track the highest score for the assigned topic
            score = 0 
            # Iterate over each sentence and calculate the score for the sentence
            for sentence in remaining_text:
                matches = sum(keyword in sentence.lower() for keyword in keywords)
                # Calculate the suitability score
                score = matches / len(keywords)  
                # Check the high score and adapt the topic
                if best_score < score:
                    best_score = score
                    best_topic = topic
            # Get the original index from the text
            id = text.index(sentence)
            # Assigned the topic to the right sentence
            assigned.append((best_topic, id))
            # Remove the sentence to iterate over the remaining sentences.
            remaining_text.remove(sentence)

        chat_title = []
        if FLAG:
            chat = Chat()
            prompt = "I’m going to send you a sentence. Please give me a title of one word that would be the most appropriate as a topic for this sentence: "
            for i , patterns in enumerate(text):
                prompt += " " + patterns
                chat.response(prompt)
                content = chat.get_result()
                chat_title.append((content , i))
                prompt = "I’m going to send you a sentence. Please give me a title of one word that would be the most appropriate as a topic for this sentence: "
        return assigned , chat_title
