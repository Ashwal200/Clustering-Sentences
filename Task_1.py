import requests
from bs4 import BeautifulSoup
import re

# Define the URL of the webpage
urls = ["https://patents.google.com/patent/GB2478972A/en?q=(phone)&oq=phone", "https://patents.google.com/patent/US9634864B2/en?oq=US9634864B2",
        "https://patents.google.com/patent/US9980046B2/en?oq=US9980046B2"]


class patenTools:
    def __init__(self, url):
        self.__url__ = url
        self.__data__ = None


    def url_reader(self):
        # Send a GET request to the URL
        keyword = 'Claims'
        response = requests.get(self.__url__)

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

            self.__data__ = keyword_sentences
        else:
            print("Failed to fetch the webpage. Status code:", response.status_code)
    def get_result(self):
        print(self.__data__)
        
if __name__ == '__main__':
    for url in urls:
        pt = patenTools(urls)
        pt.url_reader()
        pt.get_result()
