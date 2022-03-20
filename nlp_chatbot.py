import streamlit as st
import math
import os
import subprocess
from datetime import datetime
from transformers import pipeline
import stanza # https://pypi.org/project/stanza/ 
import wikipedia
import networkx as nx # https://pypi.org/project/networkx/ 
#from langdetect import detect
from deep_translator import GoogleTranslator
from pydub import AudioSegment # librairie qui permet de manipuler un fichier audio simplement (ouverture de l'audio, augmentation / diminuaiton du volume de l'audio, segmentation de l'audio, calcul de la durée de l'audio, et toute autre manipulation possible sur un audio. La librairie est en open-source sur https://github.com/jiaaro/pydub). Pour utiliser pydub, il faut installer ffmpeg/avlib.
import speech_recognition as sr # librairie (https://github.com/Uberi/speech_recognition) qui effectue de la reconnaissance vocale via diverses APIs, en ligne et hors ligne, notamment CMU Sphinx, Google Cloud Speech, WIT.ai, Microsoft Azure Speech, Houndify, IBM Speech to Text, ou encore Snowboy Hotword Detection.
import pyaudio # PyAudio fournit des liaisons Python pour PortAudio, la bibliothèque de référence qui gère les entrée/sortie usb d'audio sur toutes les plateformes (dont la raspberry). 
import wave # La librairie wave fournit une interface pour le format de son WAV. Il permet d'exporter simplement un fichier .wav.
from pydub import AudioSegment # librairie qui permet de manipuler un fichier audio simplement (ouverture de l'audio, augmentation / diminuaiton du volume de l'audio, segmentation de l'audio, calcul de la durée de l'audio, et toute autre manipulation possible sur un audio. La librairie est en open-source sur https://github.com/jiaaro/pydub). Pour utiliser pydub, il faut installer ffmpeg/avlib.

FORMAT = pyaudio.paInt16 # Le son est stocké en binaire, comme tout ce qui concerne les ordinateurs. Ici, 16 bits sont stockés par échantillon
CHANNELS = 1 # chaque trame a 1 échantillon (16 bits)
RATE = 48000 # 48000 images sont collectées par secondes. L'unité est le Hz. On a obtenu ce chiffre en faisant "p.get_device_info_by_index(0)['defaultSampleRate']" (sachant que l'index de notre usb microphone device est de 0)
# En d'autres termes, par seconde, le système lit 48000/1025 morceaux de la mémoire (soit c.4,7). Ici, cela dépend du microphone. 
FRAMES_PER_BUFFER = 1024 # nombre de trames dans lesquelles les signaux sont divisés (ce chiffre est une puissance de 2, ici 2**10)
st.set_option('deprecation.showPyplotGlobalUse', False)

class ConvertAudioToText():
    def __init__(self, language):
        '''
        Cette classe va découper l'audio en sous-audios de 10s et va appeler au fur et à mesure l'API Google de Speech Recognition pour effectuer la transcription sur chaque bout de 10s. Effectivement, l'algorithme de Google a été entraîné sur des audios dont la taille varie entre 1 seconde à 20 secondes.
        - language: fr-FR for french, en-GB for english (UK), en-US for english (us), de-DE for german, es-ES for spanish, it-IT for italian
        '''
        self.language = language
        self.frames = [] # Pyaudio ne permet pas d'enregistrer un audio en continue, mais à la place enregistre sous forme de slot. Dès lors, nous ajoutons ces slots d'audio (1 slot = 1s) dans cet array "frame".
        self.final_result = "" # stocke le résultat final (le texte de l'audio transcrit)
        self.audio = None # ouvre le fichier .wav en tant qu'instance AudioSegment à l'aide de la méthode AudioSegment.from_file() (ici, from_wav)
        # Il est important de comprendre que pyaudio découpe les données en CHUNKS (trames), au lieu d'avoir une quantité continue d'audio, afin de limiter la puissance de traitement requise (RAM), puisque la Raspberry a une RAM assez faible (512M pour notre Raspberry Pi Zero W). 
        self.WAVE_OUTPUT_FILENAME = datetime.now().strftime("audio/%d%b%Y_%Hh%Mmin.wav") # on nomme le fichier avec %d le jour d'aujourd'hui, %b le nom du mois en anglais, %Y l'année en 4 chiffres, puis _, puis l'heure suivie de h, et les minutes suivies de "min". On place l'audio dans le folder "audio_folder".
    
    def register_audio(self, stream):
        input_audio = stream.read(2 * RATE, exception_on_overflow = False) # on enregistre 1s
        self.frames.append(input_audio) # on ajoute la seconde d'enregistrement à frames (voir explication de la variable "frames")

    def save_audio_wav(self):
        wav_file = wave.open(self.WAVE_OUTPUT_FILENAME, "wb") # on ouvre ce document wav qui porte le nom ci-dessus (WAVE_OUTPUT_FILENAME) pour écrire dessus
        wav_file.setnchannels(1) # On rappelle les channels de l'audio
        wav_file.setsampwidth(2)
        wav_file.setframerate(48000) # On rappelle les Hz de l'audio
        wav_file.writeframes(b''.join(self.frames)) # on écrit dans le fichier wav ce qu'on a enregistré ("frames")
        wav_file.close() # on ferme le fichier wav.
        self.frames = [] # On vide l'enregistrement de la mémoire.
        song = AudioSegment.from_wav(self.WAVE_OUTPUT_FILENAME) # On récupère le fichier wav sauvé
        #song = song + 20 # On lui ajoute manuellement 20db (pour que le son soit plus fort)
        song.export(self.WAVE_OUTPUT_FILENAME, "wav") # On exporte de nouveau le fichier wav qui a ce coup-ci 20db de plus.
        st.text(f"audio {self.WAVE_OUTPUT_FILENAME}.wav saved!")
    
    def conversion_into_text(self, audio_file):
        '''
        - audio_file: chemin du fichier .wav exporté (ce n'est pas le fichier original, mais le fichier préprocessé qui a été sauvegardé
        '''
        r = sr.Recognizer() # instancie un objet de type Recognizer
        audio_ex = sr.AudioFile(audio_file) # nécessaire d'avoir ce format pour le Recognizer de Speech_recognition
        with audio_ex as source:
            audiodata = r.record(audio_ex) # lit l'audio
            return r.recognize_google(audiodata, language=self.language) # appelle l'API Google en utilisant la clé d'API par défaut (pas de limite d'appel)

    def get_duration(self):
        return self.audio.duration_seconds # la fonciton duration_seconds permet donner la durée en secondes d'un fichier audio à l'aide de pydub
    
    def algorithm_on_single_split_of_audio(self, from_sec, to_sec):
        '''
        Cette fonction découpe un fichier audio initial entre la seconde from_sec de l'audio et la seconde to_sec de l'audio. 
        - from_sec:
        - to_sec
        '''
        t1 = from_sec * 1000 # il faut convertir le temps de début en milisecondes
        t2 = to_sec * 1000 # il faut convertir le temps de fin en milisecondes
        split_audio = self.audio[t1-1000:t2+1000] # on récupère l'audio entre les milisecondes t1 et t2. Afin d'avoir le moins possible de mot haché lors du découpage, ce qui empêche une transcription correcte, nous prenons 1 secondes avant et après pour chaque slot d'audio (soit 1000 milisecondes avant et après)
        if split_audio.duration_seconds == 0: # Si on est en début ou fin de l'audio, il se peut que le slot de ait une durée de 0 (car on ne peut pas prendre l'audio à partir de la -1000ème miliseconde ou jusqu'à la +1000ème). Du coup, on doit gérer ces cas:
            split_audio = self.audio[t1-1000:t2] # Dans le cas où on est à la fin de l'audio (si on est au début, split_audio.duration == 0 ici)
            if split_audio.duration_seconds == 0: # Dans le cas où on est au début  de l'audio (si on est à la fin de l'audio, on a self.audio[t1-1000:t2].duration_seconds != 0 donc on ne rentre pas ici), mais où self.audio[t1-1000:t2].duration_seconds == 0 puisque self.audio[t1-1000] n'existe pas.
                split_audio = self.audio[t1:t2+1000] # Dans le cas où on est au début  de l'audio
        try:
            os.remove("audio/audio_exported.wav") # s'il y a déjà un slot d'audio exporté (voir ci-dessous: on exporte chaque slot d'audio, que l'on traite avec la fonction de speech_recognition, et que l'on supprime une fois la transcription du slot effectuée). Le try / except permet de ne pas avoir d'erreur si jamais on est au premier découpage de l'audio, par exemple.
        except:
            pass
        split_audio.export("audio/audio_exported.wav", format="wav") # On exporte le slot de 10 secondes (entre 11 et 12 exactement puisqu'on a pris une seconde de plus au début et une de plus à la fin (à part quand on est au début ou à la fin de l'audio, comme expliqué ci-dessus))
        try:
            res = self.conversion_into_text("audio/audio_exported.wav") # on appelle la fonction conversion_into_text détaillée ci-dessus qui retourne la transcription (en string) de l'audio (path du slot de l'audio: audio/audio_exported.wav, qu'on connaît puisque c'est nous qui l'avons exporté ci-dessus)
            self.final_result += " " + res # On ajoute la transcription du slot au résultat final. On a ajouté un espace entre chaque transcription de slots (" ") qui est nécessaire pour ne pas avoir 2 mots collés. 
        except:
            pass
        os.remove("audio/audio_exported.wav") # On supprime le slot une fois la transcription effectuée

    def main_conversion_answer_to_text(self):
        st.text("Starting converting audio file...")
        self.audio = AudioSegment.from_wav(self.WAVE_ANSWER_FILENAME)
        total_sec = math.ceil(self.get_duration()) # self.get_duration retourne un flottant. On veut ici itérer sur l'ensemble des slots de 10 secondes de l'audio. Si on a la durée totale de l'audio qui n'est pas égale à un nombre rond, on va prendre l'entier supérieur à ce-dernier (35.4s donne 36s), car on veut être sûr d'avoir l'intégtalité de l'audio, quitte à avoir un slot plus petit que 10 secondes à la fin. On a nécessairement besoin d'avoir un entier ici pour pouvoir itérer avec la fonction "range" ci-dessous. 
        for i in range(0, total_sec, 10): # On itère sur le nombre total de slots (de 0 à total_sec avec un saut de 10, car on veut des slots de 10s)
            self.algorithm_on_single_split_of_audio(i, i+10) # On applique l'algorithme de transcription de l'audio entre la seconde i et la seconde i+10 de l'audio.
        return self.final_result.strip() # on retourne, sous forme de string, le texte final. La fonction strip() permet d'enlever les espaces en trop au début et à la fin du texte.

    def main_conversion_audio_to_text(self):
        st.text("Starting converting audio file...")
        print("We translate the following audio : ", self.WAVE_OUTPUT_FILENAME)
        self.audio = AudioSegment.from_wav(self.WAVE_OUTPUT_FILENAME)
        total_sec = math.ceil(self.get_duration()) # self.get_duration retourne un flottant. On veut ici itérer sur l'ensemble des slots de 10 secondes de l'audio. Si on a la durée totale de l'audio qui n'est pas égale à un nombre rond, on va prendre l'entier supérieur à ce-dernier (35.4s donne 36s), car on veut être sûr d'avoir l'intégtalité de l'audio, quitte à avoir un slot plus petit que 10 secondes à la fin. On a nécessairement besoin d'avoir un entier ici pour pouvoir itérer avec la fonction "range" ci-dessous. 
        print("total sec of the audio : ", total_sec)
        for i in range(0, total_sec, 10): # On itère sur le nombre total de slots (de 0 à total_sec avec un saut de 10, car on veut des slots de 10s)
            self.algorithm_on_single_split_of_audio(i, i+10) # On applique l'algorithme de transcription de l'audio entre la seconde i et la seconde i+10 de l'audio.
        print("text here : ", self.final_result.strip())
        return self.final_result.strip() # on retourne, sous forme de string, le texte final. La fonction strip() permet d'enlever les espaces en trop au début et à la fin du texte.

def show_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels = True, labels = nx.get_node_attributes(G, "label"))
    nx.draw_networkx_edge_labels(G, pos, edge_labels = nx.get_edge_attributes(G, "label"), font_color='red')
    st.pyplot()

def build_graph(list_of_words):
    G = nx.DiGraph()
    labels = {}
    # add nodes
    for i, word in enumerate(list_of_words):
        G.add_node(i) # l'identifiant de chaque noeud doit être unique
        labels[i] = word.text
    nx.set_node_attributes(G, labels, name = "label") # tous les noeuds ont un attribut qui s'appelle label et qui contient le nom du label
    # add links between nodes (edges)
    edge_labels = {}
    for j, word in enumerate(list_of_words):
        if word.head != 0:
            G.add_edge(j, word.head - 1)
            edge_labels[(j, word.head - 1)] = word.deprel
    nx.set_edge_attributes(G, edge_labels, name = "label")
    return G

def get_qa_pipeline():
    qa = pipeline("question-answering", framework="pt")
    return qa

def answer_question(pipeline, question, paragraph):
    input = {"question": question, "context": paragraph}
    return pipeline(input)

class DefineGeneralVariables():
    def __init__(self):
        self.p = None
        self.stream = None

def start():
    st.text("starting recording your question...")

def stop():
    st.text("stopping recording...")
    defineentity.stream.stop_stream()
    defineentity.stream.close()
    defineentity.p.terminate()
    audio_converter.save_audio_wav()
    question = audio_converter.main_conversion_audio_to_text()
    answer_from_bot = str(GoogleTranslator(source='auto', target=language.split("-")[0]).translate(f"I am analyzing your question. To do so, I am plotting here the Dependency Grammar of your question asked."))
    subprocess.run(['tts', "--text", f"{answer_from_bot}", "--model_name", str(model_name), "--out_path", "audio/tts_output.wav"], 
                    stdout=subprocess.PIPE, 
                    universal_newlines=True)
    subprocess.run(["play", "audio/tts_output.wav"], stdout=subprocess.PIPE, 
                    universal_newlines=True)
    stanza_model = stanza.Pipeline(language.split("-")[0]) # This sets up a default neural pipeline in the right language
    doc = stanza_model(question)
    for sentence_number in range(len(doc.sentences)):
        G = build_graph(doc.sentences[sentence_number].words)
        show_graph(G)
    if question != "":
        if language != "en":
            question = str(GoogleTranslator(source='auto', target="en").translate(question))
        pipeline = get_qa_pipeline()
        try:
            answer = str(answer_question(pipeline, question, variables.wiki_text)["answer"])
            if answer != "":
                try:
                    answer_from_bot = str(GoogleTranslator(source='auto', target=language.split("-")[0]).translate(answer))
                except:
                    answer_from_bot = answer
                subprocess.run(['tts', "--text", f"{answer_from_bot}", "--model_name", str(model_name), "--out_path", "audio/tts_output.wav"], 
                                stdout=subprocess.PIPE, 
                                universal_newlines=True)
                subprocess.run(["play", "audio/tts_output.wav"], stdout=subprocess.PIPE, 
                                universal_newlines=True)
            else:
                answer_from_bot = str(GoogleTranslator(source='auto', target=language.split("-")[0]).translate("Sorry, I haven't found any answer for your question. Please try another question."))
                subprocess.run(['tts', "--text", f"{answer_from_bot}", "--model_name", str(model_name), "--out_path", "audio/tts_output.wav"], 
                                stdout=subprocess.PIPE, 
                                universal_newlines=True)
                subprocess.run(["play", "audio/tts_output.wav"], stdout=subprocess.PIPE, 
                                universal_newlines=True)
        except Exception as e:
            print(e)
    else:
        answer_from_bot = str(GoogleTranslator(source='auto', target=language.split("-")[0]).translate("Sorry, I haven't understood your question. Please try again."))
        subprocess.run(['tts', "--text", f"{answer_from_bot}", "--model_name", str(model_name), "--out_path", "audio/tts_output.wav"], 
                        stdout=subprocess.PIPE, 
                        universal_newlines=True)
        subprocess.run(["play", "audio/tts_output.wav"], stdout=subprocess.PIPE, 
                        universal_newlines=True)

class Variables():
    def __init__(self):
        self.entity_wiki_chosen = None
        self.wiki_text = None
        self.entity_queried = None

variables = Variables()
paragraph_slot = st.empty()
variables.entity_queried = st.text_input("Select one entity:", "")
if variables.entity_queried != None and variables.entity_queried != "":
    results = wikipedia.search(variables.entity_queried)
    list_results = [None]
    for result in results:
        list_results.append(result)
    variables.entity_wiki_chosen = st.selectbox(
    'Which entity are you talking about? Choose one from the following list: ',
    list_results)
    if variables.entity_wiki_chosen != None:
        try:
            variables.wiki_text = wikipedia.summary(variables.entity_wiki_chosen.replace(" ", "_"), sentences = 10000)
        except Exception as e:
            print(e)
            variables.wiki_text = None
    if variables.wiki_text != None:
        defineentity = DefineGeneralVariables()
        language = st.radio('Select your language:', ['None', 'fr-FR', 'en-GB', 'en-US'])
        if language != "en":
            variables.wiki_text = str(GoogleTranslator(source='auto', target="en").translate(variables.wiki_text))
        start_button = st.button("Click to record a question", on_click=start)
        stop_button = st.button("Click to stop recording", on_click=stop)
        if language != "None" and start_button:
            dico_models_text_to_speech = {"en": "tts_models/en/ljspeech/tacotron2-DDC", "fr": "tts_models/fr/mai/tacotron2-DDC"}
            model_name = dico_models_text_to_speech[language.split('-')[0]]
            audio_converter = ConvertAudioToText(language)            
            defineentity.p = pyaudio.PyAudio() # On instancie un objet Pyaudio
            defineentity.stream = defineentity.p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER)
            while not stop_button:
                st.text("Recording...")
                audio_converter.register_audio(defineentity.stream)