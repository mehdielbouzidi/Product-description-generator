import argparse
import os
from warnings import warn
from time import sleep
from flask import Flask
from flask import request, redirect, render_template
from flask import session
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)


def generate_description(seed_text, next_words, model, max_seq_len):

	# loading
    with open('./static/tokenizer.pickle', 'rb') as handle:
        t = pickle.load(handle)

    for _ in range(next_words):
    
        token_list = t.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ''
        
        for word,index in t.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        seed_text = seed_text + " " + output_word
        
    return seed_text


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        req = request.form
        model=req['model']
        purpose=req['purpose']
        length=req['length']
        start_seq=model+purpose
        model = load_model('./static/my_model.h5')
        output=(generate_description(start_seq, int(length), model, 155))
        return render_template('index.html', prediction_text=output)

    else:

        return render_template("index.html")

# return "Hello world!"


    
if __name__ == '__main__':
	
	app.run()

    
