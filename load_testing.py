import pandas as pd
from locust import HttpUser, task, FastHttpUser
import random

emotions = pd.read_csv(
    "dataset\Goemotions_team_v2_reduced.csv")

sentences = list(emotions['text'])


class WebsiteUser(FastHttpUser):
    host = "https://jeevavijay10-nlp-goemotions-senti-pred.hf.space"
    
    def wait_time(self):
        return 240
    @task
    def predict(self):
        sentence = random.choice(sentences)
        # print(sentence)

        data = {"data": [sentence]}
        resp = self.client.post("/run/predict", json=data)        
        # print(resp.json())
        return resp
