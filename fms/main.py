from flask import Flask
from flask import request
from flask.logging import default_handler
import logging as log
import signal
import sys
import cv2 as cv
import os
import gc
import numpy as np
import argparse
import json
import csv
import multiprocessing
import face_recognition
from queue import Queue

# local modules
import config
import face_encoding

# globals
known_people = []
known_encodings = []

app = Flask(__name__)

def graceful_shutdown():
	log.info('Gracefully shutting down %s ...', config.app_name)
	sys.exit(0)

def signal_handler(sig, frame):
	log.debug("%s received", signal.Signals(2).name)
	log.debug("Attempting to initiate graceful shutdown ...")
	graceful_shutdown()

@app.route('/', methods=['GET'])
def get_index():
	return str(config.app_name) + " is running"

@app.route('/search', methods=['POST'])
def post_search():
    
    if not os.path.exists(config.image_output_directory):
        os.makedirs(config.image_output_directory)

    image_file = request.files['image']
    image_file_buf = image_file.read()
    npimg = np.fromstring(image_file_buf, np.uint8)
    image = cv.imdecode(npimg, cv.IMREAD_COLOR)

    log.debug("Processing image %s" % (image_file.filename))

    encodings = face_encoding.process([image])
    matched_people = []

    for encoding in encodings:

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if True in matches:
            first_match_index = matches.index(True)
            matched_people.append(known_people[first_match_index])
    
    return {
        "success": True,
        "total": encodings.count,
        "match": matched_people.count,
        "noMatch": total - matched_people.count,
        "results": matched_people
    }

@app.route('/enroll', methods=['POST'])
def post_enroll():
    
    if not os.path.exists(config.image_output_directory):
        os.makedirs(config.image_output_directory)

    image_file = request.files['image']
    image_file_buf = image_file.read()
    npimg = np.fromstring(image_file_buf, np.uint8)
    image = cv.imdecode(npimg, cv.IMREAD_COLOR)

    encodings = face_encoding.process([image])
    if(encodings.count != 1):
        return { "success": False }

    person_name = request.json.name
    
    save_enrollment([person_name, encodings[0]])
    known_people.append(person_name)
    known_encodings.append(encodings[0])

    log.debug("Enrolled (%s, %s)" % (person_name, encodings[0]))

    return { "success": True }

def load_enrollments(file_name):
    
    global known_people
    global known_encodings

    f = open(file_name, 'w+')

    known_people = []
    known_encodings = []

    reader = csv.reader(f, delimiter=' ', quotechar='|')
    for row in reader:
        known_people.append(row[0])
        known_encodings.append(row[1])
    
    f.close()

    return

def save_enrollment(enrollment):
    
    with open("data/enrollments.csv",'a') as fd:
        fd.write(enrollment)

    return

def print_banner(app_name, app_version):
	spaced_text = " " + str(app_name) + " " + str(app_version) + " "
	banner = spaced_text.center(78, '=')
	filler = ''.center(78, '=')
	log.info(filler)
	log.info(banner)
	log.info(filler)

def main():

    global app

    signal.signal(signal.SIGINT, signal_handler)

    # set path to main.py path
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description='Entrypoint script for Checkingo Face Matching Service (FMS)'
    )
    parser.add_argument(
        '-f',
        '--config_file',
        help='Path to configuration file.',
        default='config/app.yaml'
    )
    args = parser.parse_args()

    config.load(args.config_file)

    load_enrollments("data/enrollments.csv")

    print_banner(config.app_name, config.app_version)

    log.info(str(config.app_name) + " is running on port " + str(config.flask_port))

    logger = log.getLogger('werkzeug')
    logger.setLevel(log.ERROR)
    app.run(host='127.0.0.1', port=config.flask_port, debug=config.flask_debug)

if __name__ == "__main__":
	main()