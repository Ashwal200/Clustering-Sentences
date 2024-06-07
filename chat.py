#!/usr/bin/env -S poetry run python

from openai import OpenAI
import time
# Set your API key here
api_key = ""

# Initialize the client with the API key
client = OpenAI(api_key=api_key)
class Chat:
    def __init__(self):
        self.__content__ = None
    print("----- standard request -----")
    def response(self, prompt):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        self.__content__ = completion.choices[0].message.content.strip()

    def get_result(self):
        return self.__content__
