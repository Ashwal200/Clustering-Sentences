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
        self.__train__ = None
        self.__test__ = None
        self.__data__ = None
        

    def print_the_results(self, labels, sentences):
        cluster_dict = {}
        for cluster_num, sentence in zip(labels, sentences):
            if cluster_num not in cluster_dict:
                cluster_dict[cluster_num] = [sentence]
            else:
                cluster_dict[cluster_num].append(sentence)

        all_text = []
        array_num_of_claims = []
        text = ""
        for cluster_num, cluster_sentences in cluster_dict.items():
            number_of_claims = 0
            for sentence in cluster_sentences:
                number_of_claims += 1
                text += " " + sentence
            all_text.append(text)
            array_num_of_claims.append(number_of_claims)
            text = ""
        data = []
        titles , chat_title = self.topic_render(all_text)
        for title in titles:
            data.append(f"title: {title[0]}, numbers of claims: {array_num_of_claims[title[1]-1]}")
        
        if FLAG:
            data.append(f"---> Also the chat provide titles:")
            # Convert new data tuples to the desired format and append them to the existing list
            for title, count in chat_title:
                data.append(f"title: {title}, numbers of claims: {array_num_of_claims[count]}")
        self.__data__ = data

    def url_reader(self, url):
        # Send a GET request to the URL
        keyword = 'Claims'
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, "html.parser")
            
            topic_title = soup.title.string

            components = topic_title.split(" - ")

            title = components[1]

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
                            "suggest", "state", "believe", "affirm", "maintain", "contend", "insist" , "apparatus"]
            
            # List to store sentences containing the keyword
            keyword_sentences = []
            # Iterate through the sentences
            for sentence in sentences:
                # Check if the keyword is found in the sentence
                if keyword in sentence:
                    # Set the flag to True to start adding sentences
                    add_sentences = True


                # Add the sentence if the flag is True and it's not empty
                if add_sentences and any(word.lower() in sentence.lower() for word in assertion_words) and sentence.strip().endswith(".") and title not in sentence and type_patent not in sentence and "Fig" not in sentence and "Claims (" not in sentence:
                    keyword_sentences.append(sentence.strip())

            return keyword_sentences
        else:
            print("Failed to fetch the webpage. Status code:", response.status_code)

    def split_test_train(self):
        train_claim_sentences = []
        test_claim_sentences = []

        # Data Collection and Preprocessing
        for i, url in enumerate(self.__urls__):
            # Fetch and process the text from the URL
            text = self.url_reader(url)
            if i == 2:
                test_claim_sentences = text
            else:
                train_claim_sentences.extend(text)

        self.__train__ = train_claim_sentences 
        self.__test__ = test_claim_sentences
    
    
   # Convert each sentence to a vector representation by averaging word vectors
    def sentence_to_vector(self,sentence, model):
        words = [word for word in sentence.split() if word in model.wv]
        if not words:
            return np.zeros(model.vector_size)
        vectors = [model.wv[word] for word in words]
        return np.mean(vectors, axis=0)

    def test_model_KMeansW2V(self , kmeans_model , test_claim_sentences , word2vec_model):
        new_data_vectors = np.array([self.sentence_to_vector(sentence, word2vec_model) for sentence in test_claim_sentences])
        if len(new_data_vectors) > 0:
            new_data_clusters = kmeans_model.predict(new_data_vectors)
            
            # Calculate distances to cluster centers for probability-like measure
            distances = kmeans_model.transform(new_data_vectors)

            self.print_the_results(new_data_clusters, test_claim_sentences)
        
    
    def model_KMeansW2V(self, num_clusters):
        self.split_test_train()
        train_claim_sentences = self.__train__
        test_claim_sentences = self.__test__
        # Train Word2Vec model
        word2vec_model = Word2Vec(sentences=[sentence.split() for sentence in train_claim_sentences], vector_size=100, window=5, min_count=1, workers=4)

        train_vectors = np.array([self.sentence_to_vector(sentence, word2vec_model) for sentence in train_claim_sentences])

        # Topic Modeling (K-means clustering)
        kmeans_model = KMeans(n_clusters=int(num_clusters), random_state=42)
        kmeans_model.fit(train_vectors)

        self.test_model_KMeansW2V(kmeans_model , test_claim_sentences , word2vec_model)




    def get_result(self):
        return self.__data__
    
    def topic_render(self,text):
        # Define keywords for each topic
        topic_keywords = {
            "Wireless": ["wireless", "loss"],
            "Telephone": ["telephone", "phone"],
            "Call": ["call", "indication" , "conversation"],
            "Functionality": ["functionality","Efficient" , "switching" , "quality of service"],
            "Communication": ["communication", "source" , "network"],
            "Device": ["device"]
        }

        # Assign topics to sentences
        titles , chat_titles = self.assign_topic(text , topic_keywords)
        
        return titles , chat_titles


# Function to assign topic to a sentence based on occurrence of keywords
    def assign_topic(self,sentences, topic_keywords):
        assigned = []
        remaining_sentences = sentences.copy()  
        # Iterate over each sentence
        for topic, keywords in topic_keywords.items():
            if not remaining_sentences:
                break
            best_topic = "Other"
            best_score = 0 # Track the highest score for the assigned topic
            score = 0 
            # Iterate over each topic and calculate the score for the sentence
            for sentence in remaining_sentences:
                matches = sum(keyword in sentence.lower() for keyword in keywords)
                score = matches / len(keywords)  # Calculate the suitability score
                if best_score < score:
                    best_score = score
                    best_topic = topic
            id = sentences.index(sentence)
            assigned.append((best_topic, id))
            remaining_sentences.remove(sentence)

        chat_title = []
        if FLAG:
            chat = Chat()
            prompt = "I’m going to send you a sentence. Please give me a title of one word that would be the most appropriate as a topic for this sentence: "
            for i , text in enumerate(sentences):
                prompt += " " + text
                chat.response(prompt)
                content = chat.get_result()
                chat_title.append((content , i))
                prompt = "I’m going to send you a sentence. Please give me a title of one word that would be the most appropriate as a topic for this sentence: "
        return assigned , chat_title
