# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from flask_cors import CORS
import re


def dog_names():
    dog_breed_names = ['n02085620-Chihuahua',
    'n02085782-Japanese_spaniel',
    'n02085936-Maltese_dog',
    'n02086079-Pekinese',
    'n02086240-Shih-Tzu',
    'n02086646-Blenheim_spaniel',
    'n02086910-papillon',
    'n02087046-toy_terrier',
    'n02087394-Rhodesian_ridgeback',
    'n02088094-Afghan_hound',
    'n02088238-basset',
    'n02088364-beagle',
    'n02088466-bloodhound',
    'n02088632-bluetick',
    'n02089078-black-and-tan_coonhound',
    'n02089867-Walker_hound',
    'n02089973-English_foxhound',
    'n02090379-redbone',
    'n02090622-borzoi',
    'n02090721-Irish_wolfhound',
    'n02091032-Italian_greyhound',
    'n02091134-whippet',
    'n02091244-Ibizan_hound',
    'n02091467-Norwegian_elkhound',
    'n02091635-otterhound',
    'n02091831-Saluki',
    'n02092002-Scottish_deerhound',
    'n02092339-Weimaraner',
    'n02093256-Staffordshire_bullterrier',
    'n02093428-American_Staffordshire_terrier',
    'n02093647-Bedlington_terrier',
    'n02093754-Border_terrier',
    'n02093859-Kerry_blue_terrier',
    'n02093991-Irish_terrier',
    'n02094114-Norfolk_terrier',
    'n02094258-Norwich_terrier',
    'n02094433-Yorkshire_terrier',
    'n02095314-wire-haired_fox_terrier',
    'n02095570-Lakeland_terrier',
    'n02095889-Sealyham_terrier',
    'n02096051-Airedale',
    'n02096177-cairn',
    'n02096294-Australian_terrier',
    'n02096437-Dandie_Dinmont',
    'n02096585-Boston_bull',
    'n02097047-miniature_schnauzer',
    'n02097130-giant_schnauzer',
    'n02097209-standard_schnauzer',
    'n02097298-Scotch_terrier',
    'n02097474-Tibetan_terrier',
    'n02097658-silky_terrier',
    'n02098105-soft-coated_wheaten_terrier',
    'n02098286-West_Highland_white_terrier',
    'n02098413-Lhasa',
    'n02099267-flat-coated_retriever',
    'n02099429-curly-coated_retriever',
    'n02099601-golden_retriever',
    'n02099712-Labrador_retriever',
    'n02099849-Chesapeake_Bay_retriever',
    'n02100236-German_short-haired_pointer',
    'n02100583-vizsla',
    'n02100735-English_setter',
    'n02100877-Irish_setter',
    'n02101006-Gordon_setter',
    'n02101388-Brittany_spaniel',
    'n02101556-clumber',
    'n02102040-English_springer',
    'n02102177-Welsh_springer_spaniel',
    'n02102318-cocker_spaniel',
    'n02102480-Sussex_spaniel',
    'n02102973-Irish_water_spaniel',
    'n02104029-kuvasz',
    'n02104365-schipperke',
    'n02105056-groenendael',
    'n02105162-malinois',
    'n02105251-briard',
    'n02105412-kelpie',
    'n02105505-komondor',
    'n02105641-Old_English_sheepdog',
    'n02105855-Shetland_sheepdog',
    'n02106030-collie',
    'n02106166-Border_collie',
    'n02106382-Bouvier_des_Flandres',
    'n02106550-Rottweiler',
    'n02106662-German_shepherd',
    'n02107142-Doberman',
    'n02107312-miniature_pinscher',
    'n02107574-Greater_Swiss_Mountain_dog',
    'n02107683-Bernese_mountain_dog',
    'n02107908-Appenzeller',
    'n02108000-EntleBucher',
    'n02108089-boxer',
    'n02108422-bull_mastiff',
    'n02108551-Tibetan_mastiff',
    'n02108915-French_bulldog',
    'n02109047-Great_Dane',
    'n02109525-Saint_Bernard',
    'n02109961-Eskimo_dog',
    'n02110063-malamute',
    'n02110185-Siberian_husky',
    'n02110627-affenpinscher',
    'n02110806-basenji',
    'n02110958-pug',
    'n02111129-Leonberg',
    'n02111277-Newfoundland',
    'n02111500-Great_Pyrenees',
    'n02111889-Samoyed',
    'n02112018-Pomeranian',
    'n02112137-chow',
    'n02112350-keeshond',
    'n02112706-Brabancon_griffon',
    'n02113023-Pembroke',
    'n02113186-Cardigan',
    'n02113624-toy_poodle',
    'n02113712-miniature_poodle',
    'n02113799-standard_poodle',
    'n02113978-Mexican_hairless',
    'n02115641-dingo',
    'n02115913-dhole',
    'n02116738-African_hunting_dog']

    dog_names = [re.sub(r'^n\d+-', '', name) for name in dog_breed_names]

    return dog_names


app = Flask(__name__)
CORS(app)
model = load_model('dog_model.keras')


def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image)
    image_array = image_array / 255.0  
    image_array = np.expand_dims(image_array, axis=0)  
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file)
        image = preprocess_image(image)

        prediction = model.predict(image)
        predicted_class_idx = np.argmax(prediction)
        dog_names_list = dog_names()
        predicted_class_name = dog_names_list[predicted_class_idx]
        print(predicted_class_name)

        response = {
            'prediction': str(predicted_class_name)  
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
