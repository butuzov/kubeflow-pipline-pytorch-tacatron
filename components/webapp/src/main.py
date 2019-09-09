from network import *
from dataload import inv_spectrogram, find_endpoint, save_wav, spectrogram
import numpy as np
import argparse
import os, sys
import io
from text import text_to_sequence
from librosa.output import write_wav

from flask import Flask, session, url_for, render_template, redirect, request, send_from_directory
import hashlib
# helper for default text
import dune

# debug hlper
import json

# IS_CUDA = torch.cuda.is_available()

APP = Flask(__name__)


@APP.route("/", methods=['GET', 'POST'])
def home():
    """index page and generator"""

    args = parser.parse_args()

    text  = dune.quote()     # Default palceholder text
    error = None        # Errors
    ok    = None        # Ok status
    gen   = None




    # model
    model = get_model(os.path.join(args.directory, args.model))

    # application logic
    if request.method == 'POST':

        if request.form.get("quote", None):
            gen   = True
            text  = dune.quote()
            ok    = True

        elif len(request.form["tts"]) < 3:
            gen   = False
            text  = dune.quote()
            error = "Please enter some text. While you think on the content, we have added some wise words from the book `Duna`."
        else:
            text  = request.form["tts"]
            ok    = True
            gen   = True

    name = None
    if gen and ok and text:
        wav = generate(model, text)
        name_encoder = hashlib.md5()
        name_encoder.update(str.encode(text))

        # Make checkpoint directory if not exists
        if not os.path.exists(args.results):
            os.mkdir(args.results)

        name = str(name_encoder.hexdigest())
        path = os.path.join(args.results, f'{name}.wav')
        with open(path, 'wb') as f:
            f.write(wav)


    return render_template('index.html.j2', data={
        'text' : text,
        'error': error,
        'ok'   : ok,
        'name' : name,
    })


@APP.route('/results/<path:path>')
def send_wav(path):
    """serving wav files"""
    args = parser.parse_args()
    return send_from_directory(args.results, path)



def get_model(path_to_model):
    """return model"""
    model = nn.DataParallel(Tacotron().cuda()) if torch.cuda.is_available() else Tacotron()
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    return model


def generate(model, text):
    # Text to index sequence
    cleaner_names = [x.strip() for x in hp.cleaners.split(',')]
    seq = np.expand_dims(
            np.asarray(
                text_to_sequence(text, cleaner_names), dtype=np.int32), axis=0)

    # Provide [GO] Frame
    mel_input = np.zeros([seq.shape[0], hp.num_mels, 1], dtype=np.float32)

    # Variables
    torched = torch.cuda if torch.cuda.is_available() else torch

    characters = Variable(torch.from_numpy(seq).type(torched.LongTensor))
    mel_input = Variable(torch.from_numpy(mel_input).type(torched.FloatTensor))

    if torch.cuda.is_available():
        characters.cuda()
        mel_input.cuda()


    # Spectrogram to wav
    with torch.no_grad():
        _, linear_output = model.forward(characters, mel_input)

    wav = inv_spectrogram(linear_output[0].data.cpu().numpy())
    wav = wav[:find_endpoint(wav)]

    out = io.BytesIO()
    write_wav(out, wav, 20000)

    return out.getvalue()


if __name__ == '__main__':

    # getting data from the outside of the docker container
    from collections import namedtuple

    Arg = namedtuple('Param' , 'name type help default')

    args = [
       Arg("directory",  str, "Models Directory", "/data/models"),
       Arg("model", str, "Model to use", "model.pth.tar"),
       Arg("results", str, "Results Directory", "/data/results"),
    ]


    parser = argparse.ArgumentParser(description="Training")
    for a in args:
        parser.add_argument(f"--{a.name}", type=a.type, help=a.help, default=a.default)


    # runnig flask app
    sys.exit(APP.run(debug=True, host='0.0.0.0'))
