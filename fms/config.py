import logging as log
import yaml
import os
import cv2 as cv
import json
import sys
import time

# application parameters
app_name = ""
app_version = "0.0.0"
flask_port = 226
flask_debug = True
dlib_frame_resize_enabled = False
dlib_frame_resize_scale = 1
dlib_model = "hog"
image_enabled = False
image_output_directory = "../data"
image_type = "png"
image_jpg_quality = 95
image_png_compression = 3
image_ppm_binary_format_flag = 1
image_pgm_binary_format_flag = 1
image_pbm_binary_format_flag = 1
cv_image_params = []
services_mongodb = "http://localhost:443"
services_elasticsearch = ""
services_redis = "redis://localhost:6379"

# constants
PROGRAM_START_TIMESTAMP = time.time()

def set_logging(log_level="INFO", log_file="app.log", log_timestamp=True):

    if (log_level == "DEBUG"):
        log_level = log.DEBUG
    elif (log_level == "INFO"):
        log_level = log.INFO
    elif (log_level == "WARNING"):
        log_level = log.WARNING
    elif (log_level == "ERROR"):
        log_level = log.ERROR
    elif (log_level == "CRITICAL"):
        log_level = log.CRITICAL
    else:
        print("ERROR: invalid log level was specified log_level=" + str(log_level))
        exit(0)

    root = log.getLogger()
    root.setLevel(log_level)
    formatter = log.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

    streamHandler = log.StreamHandler(sys.stdout)
    streamHandler.setLevel(log_level)
    streamHandler.setFormatter(formatter)
    root.addHandler(streamHandler)

    if(log_timestamp):
        index = log_file.rfind(".")
        log_file_path = log_file[:index] + "." + str(int(PROGRAM_START_TIMESTAMP)) + log_file[index:]

    os.makedirs("log", exist_ok=True)
    fileHandler = log.FileHandler(log_file_path)
    fileHandler.setLevel(log_level)
    fileHandler.setFormatter(formatter)
    root.addHandler(fileHandler)

    return

def load(config_file_name):

    global app_name
    global app_version
    global extraction_layers
    global flask_port
    global flask_debug
    global dlib_frame_resize_enabled
    global dlib_frame_resize_scale
    global dlib_model
    global image_enabled
    global image_output_directory
    global image_type
    global image_jpg_quality
    global image_png_compression
    global image_ppm_binary_format_flag
    global image_pgm_binary_format_flag
    global image_pbm_binary_format_flag
    global cv_image_params
    global services_mongodb
    global services_elasticsearch
    global services_redis

    loaded_config = None
    with open(config_file_name, "r") as config_file:
        try:
            loaded_config = yaml.safe_load(config_file)
        except yaml.YAMLError as e:
            log.error(e)

    logging_config = loaded_config["logging"]
    log_enabled = bool(logging_config["enabled"])
    log_level = str(logging_config["level"])
    log_file = str(logging_config["file"])
    log_timestamp = bool(logging_config["timestamp"])
    if log_enabled:
        set_logging(log_level, log_file, log_timestamp)

    log.info("Loading config ...")

    app_name = str(loaded_config["name"])
    app_version = str(loaded_config["version"])

    flask_config = loaded_config["flask"]
    flask_port = int(flask_config["port"])
    flask_debug = bool(flask_config["debug"])

    services_config = loaded_config["services"]
    services_mongodb = str(services_config["mongodb"])
    services_elasticsearch = str(services_config["elasticsearch"])
    services_redis = str(services_config["redis"])

    image_config = loaded_config["image"]
    image_enabled = bool(image_config["enabled"])
    image_output_directory = os.path.join(
        str(image_config["output_directory"]), 
        str(int(PROGRAM_START_TIMESTAMP))
    )
    image_type =  str(image_config["type"])
    image_jpg_quality = int(image_config["jpg"]["quality"])
    image_png_compression = int(image_config["png"]["compression"])
    if image_enabled:

        os.makedirs(image_output_directory, exist_ok=True)

        if image_type == "jpg":
            cv_image_params = [int(cv.IMWRITE_JPEG_QUALITY), image_jpg_quality]
        elif image_type == "png":
            cv_image_params = [int(cv.IMWRITE_PNG_COMPRESSION), image_png_compression]
        elif image_type == "ppm":
            cv_image_params = [int(cv.IMWRITE_PXM_BINARY), image_ppm_binary_format_flag]
        elif image_type == "pgm":
            cv_image_params = [int(cv.IMWRITE_PXM_BINARY), image_pgm_binary_format_flag]
        elif image_type == "pbm":
            cv_image_params = [int(cv.IMWRITE_PXM_BINARY), image_pbm_binary_format_flag]
        else:
            log.error("Invalid image type: %s" % (image_type))
            exit(0)

    algorithm_config = loaded_config["algorithm"]

    dlib_config = algorithm_config["dlib"]
    frame_resize_config = dlib_config["frame_resize"]
    dlib_frame_resize_enabled = bool(frame_resize_config["enabled"])
    dlib_frame_resize_scale = float(frame_resize_config["scale"])
    dlib_model = str(dlib_config["model"])

    log.info("\n" + json.dumps(loaded_config, indent=4))
    log.info("Config loading complete.\n")

    return
