from pymongo import MongoClient  # Chữ C phải viết HOA

client = MongoClient("mongodb://192.168.8.212:27637/")

db = client['Voice_call_data'] #this is db (db contains collections (like table in SQL) --> contain data)

audio_collection = db["audio"]

cvs_collection = db["conversation"]

def save_to_log(chat_template):
    cvs_collection.insert_one(chat_template)
    
    
    
def save_audio_to_log(audio_template):
    audio_collection.insert_one(audio_template)
