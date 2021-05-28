from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pytz
from msd.apitest import test
from google.cloud import storage
from msd.SpeakerDiarizer import SpeakerDiarizer
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    
    return {
        "greetings" : "This is our root endpoint!"
    }
    

@app.get("/getfile")
def get_file(bucket_name, file_name, destination):
    
    storage_client = storage.Client()
    
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(file_name)
    local_file = blob.download_to_filename(destination)
    
    return local_file
    
@app.get("/diarize")
def diarize(bucket_name, file_name, destination):
    
    get_file(bucket_name, file_name, destination)
    
    input_file = destination
    
    diarizer = SpeakerDiarizer()
    
    diarizer.load(input_file)
    
    diarizer.generate_dvectors()
    diarizer.spectral_clustering()
    
    speaker_label = pd.DataFrame(diarizer.speaker_label)
    
    test_output = json.loads("""[{"speaker": "B", "start": 0.43034375, "end": 3.06284375, "text": "It was this observation that drew from Douglas."}, 
{"speaker": "B", "start": 3.35984375, "end": 5.50803125, "text": "Not immediately, but later in the evening."}, 
{"speaker": "B", "start": 5.86409375, "end": 11.566156249999999, "text": "Reply that had the interesting consequences to which I call attention. He wore blue silk stockings."},
{"speaker": "A", "start": 11.878343749999999, "end": 13.726156249999999, "text": "Blue knee pants with gold buckles."},
{"speaker": "A", "start": 14.01471875, "end": 17.64790625, "text": "A blue ruffled waist and a jacket of bright blue braided with gold."},
{"speaker": "B", "start": 18.01915625, "end": 21.009406249999998, "text": "Someone else told a story not particularly effective."},
{"speaker": "A", "start": 21.009406249999998, "end": 25.36484375, "text": "Which I saw he was not following. His hat had a peak crown at a flat brim."},
{"speaker": "A", "start": 25.653406249999996, "end": 31.18840625, "text": "And around the brim with a row of tiny golden bells that tinkled when he moved, cried one of the women."},
{"speaker": "B", "start": 31.401031250000003, "end": 34.29846875, "text": "He took no notice of her. He looked at me, but as if."},
{"speaker": "B", "start": 34.772656250000004, "end": 37.40853125, "text": "Instead of me, he saw what he spoke of."},
{"speaker": "A", "start": 37.40853125, "end": 41.31846875, "text": "Instead of shoes, the old man wore boots with turnover tops."},
{"speaker": "A", "start": 41.59184375, "end": 44.40490625, "text": "And his blue coat had wide cuffs of gold braid."},
{"speaker": "B", "start": 44.68840625, "end": 46.70328125, "text": "There was a unanimous groan at this."},
{"speaker": "B", "start": 47.174093750000004, "end": 48.32328125, "text": "And much reproach."},
{"speaker": "B", "start": 48.85653125, "end": 52.39184375, "text": "After which in his preoccupied way he explained for along."},
{"speaker": "A", "start": 52.39184375, "end": 56.07565625, "text": "Time he had wished to explore the beautiful land of Oz in which they live."}, 
{"speaker": "B", "start": 56.46715625, "end": 57.729406250000004, "text": "The story is written."},
{"speaker": "A", "start": 57.96565625, "end": 61.66128125, "text": "When they were outside tongue simply latched the door and started up the path."},
{"speaker": "B", "start": 62.13546875, "end": 64.83040625000001, "text": "I could write to my man and enclosed the key."},
{"speaker": "B", "start": 65.29278124999999, "end": 67.40721875, "text": "He could send down the packet as he finds it."},
{"speaker": "A", "start": 67.70590625, "end": 69.59084375, "text": "No one would disturb their little house."},
{"speaker": "A", "start": 69.89965624999999, "end": 73.57503125, "text": "Even if anyone came so far into the thick forest while they were gone."},
{"speaker": "B", "start": 74.02728124999999, "end": 75.71646874999999, "text": "The others resented postponement."},
{"speaker": "B", "start": 76.04553125000001, "end": 78.03509374999999, "text": "But it was just his scruples that charmed me."}, 
{"speaker": "A", "start": 78.40971875, "end": 83.06215624999999, "text": "At the foot of the mountain that separated the country of the Munchkins, from the country of the Gilligan\'s."}, 
{"speaker": "A", "start": 83.39459375000001, "end": 84.43240625000001, "text": "The path divided."}, 
{"speaker": "B", "start": 84.67709375000001, "end": 86.32578125, "text": "To this his answer was prompt."},
{"speaker": "B", "start": 86.51984375, "end": 89.79190625000001, "text": "Oh thank God no. And is the record yours."},
{"speaker": "A", "start": 89.79190625000001, "end": 92.64040625000001, "text": "He knew it would take them to the House of the Crooked Magician."}, {"speaker": "A", "start": 92.99309375000001, "end": 95.56653125, "text": "Whom he had never seen, but who was their nearest neighbor?"}, {"speaker": "B", "start": 95.92259375, "end": 98.55171874999999, "text": "He hung fire again, a womans."}, {"speaker": "A", "start": 98.55171874999999, "end": 104.26559375, "text": "All the morning they trudged up the mountain path and at new Nunca know Joe sat on a fallen tree trunk."}, {"speaker": "A", "start": 104.62503125, "end": 107.86334375000001, "text": "And ate the last of the bread which the old munchkin had placed in him."}, {"speaker": "B", "start": 107.86334375000001, "end": 110.47728125, "text": "Pocket she has been dead these 20 years."}, {"speaker": "A", "start": 110.73546875, "end": 112.11078125, "text": "Then they started on again."}, {"speaker": "A", "start": 112.40271874999999, "end": 114.40915625, "text": "And two hours later came insight."}, {"speaker": "A", "start": 114.75509374999999, "end": 116.24853125000001, "text": "Of the House of Doctor Pipt."}, {"speaker": "B", "start": 116.69065624999999, "end": 119.02953124999999, "text": "She sent me the pages in question before she died."}, {"speaker": "A", "start": 119.37715625000001, "end": 122.73696874999999, "text": "Uncle knocked at the door of the house into Chubby, Pleasant faced woman."}, {"speaker": "A", "start": 122.99178125, "end": 124.24390625000001, "text": "Dressed all in blue."}, {"speaker": "A", "start": 124.44303124999999, "end": 125.08428125, "text": "Opened it"}, {"speaker": "A", "start": 125.34584375, "end": 127.20040624999999, "text": "And greeted the visitors with a smile."}, {"speaker": "B", "start": 127.55309374999999, "end": 130.82178125, "text": "She was the most agreeable woman I\'ve ever known in her position."}, {"speaker": "B", "start": 131.07996875, "end": 133.02228125, "text": "She would have been worthy of any whatever."}, {"speaker": "A", "start": 133.37328125, "end": 136.70946874999998, "text": "I am my dear and all strangers are welcome to my home."}, {"speaker": "B", "start": 137.00646874999998, "end": 141.49353125, "text": "And simply that she said so, but that I knew she hadn\'t. I was sure I could see."}, {"speaker": "A", "start": 141.82596875, "end": 144.29984374999998, "text": "We have come from a far, lonelier place than this."}, {"speaker": "B", "start": 145.09296874999998, "end": 149.92934375, "text": "A lonelier place. You\'ll easily judge why when you hear because the thing had been such a scare."}, {"speaker": "B", "start": 150.37315625, "end": 151.66578124999998, "text": "He continued to fix me."}, {"speaker": "A", "start": 152.07078124999998, "end": 154.23246874999998, "text": "And you must be Ojo, the unlucky."}, {"speaker": "A", "start": 154.49740624999998, "end": 155.14878124999998, "text": "She added."}, {"speaker": "B", "start": 155.35465625, "end": 156.93246875, "text": "You are a cute."}, {"speaker": "A", "start": 157.51128125, "end": 160.45428124999998, "text": "JoJo had never eaten such a fine meal in all his life."}, {"speaker": "B", "start": 160.83396875, "end": 163.14415624999998, "text": "He quitted the fire and dropped back into his chair."}, {"speaker": "A", "start": 163.38884375, "end": 165.25015625, "text": "We\'re traveling. Reply Dojo."}, {"speaker": "A", "start": 165.51340625, "end": 168.58971875, "text": "And we stopped at your house just to rest and refresh ourselves."}, {"speaker": "B", "start": 168.85296875, "end": 170.71090625, "text": "Probably not till the 2nd post."}, {"speaker": "A", "start": 170.90496875, "end": 172.39165624999998, "text": "The woman seemed thoughtful."}, {"speaker": "B", "start": 172.39165624999998, "end": 174.99209374999998, "text": "It was almost the tone of hope everybody will stay."}, {"speaker": "A", "start": 175.17434375, "end": 176.99853124999998, "text": "At one end stood a great fireplace."}, {"speaker": "A", "start": 177.19259375, "end": 180.07484374999999, "text": "In which a blue log was blazing with a blue flame."}, {"speaker": "A", "start": 180.21490624999998, "end": 182.89296875, "text": "And over the fire hung four kettles in a row."}, {"speaker": "A", "start": 183.20853125, "end": 185.40903125, "text": "All bubbling and steaming at a great rate."}, {"speaker": "B", "start": 185.86971875, "end": 188.40940625, "text": "Cried the ladies whose dip Archer had been fixed."}, {"speaker": "A", "start": 188.40940625, "end": 197.39196875, "text": "It takes me several years to make this magic powder, but at this moment I am pleased to say it is nearly done. You see, I am making it for my Good Wife Margo lot."}, {"speaker": "A", "start": 197.66365625, "end": 200.05315625, "text": "Who wants to use some of it for a purpose of her own?"}, {"speaker": "B", "start": 200.34003124999998, "end": 203.66440624999998, "text": "Mrs. Griffin, however, expressed the need for a little more light you."}, {"speaker": "A", "start": 203.66440624999998, "end": 208.10421875, "text": "Must know said Margo Lot when they were all seated together on the broad window seat."}, 
{"speaker": "A", "start": 208.66953125, "end": 208.78765625, "text": ""}]""")
    
    print(speaker_label)

    return test_output