import pymongo

#url ='mongodb+srv://nikhilsuresh86:pWL78P1na7zmFPGA@cluster0.k9ykt3c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

url = "mongodb+srv://nikhilsdfghj:sdfghj@cluster0.k9ykt3c.mongodb.net/?hjgf=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(url)

db = client['OneLPDB']
# Create a new client and connect to the server
client = MongoClient(url, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
    
