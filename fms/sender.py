import os
import requests
import threading
import logging as log

# local modules
import config

HTTP_HEADERS = {'Content-type': 'application/json'}

def send_request(host, data, headers):

    response = requests.post(host, data=data, headers=headers)
    if response.status_code == 200:
        log.debug("Frame successfully sent to firm-ds")
    else:
        log.error("Error occured when trying to send frame to firm-ds, status code: %s", response.status_code)

    return

def send_image(image_file_name):

    log.debug("Sending image %s to firm-fms ...", image_file_name)

    image_file_path = os.path.join(config.image_output_directory, image_file_name)
    with open(image_file_path, 'rb') as img_file:
        files = {'image': img_file}
        data = {'name': image_file_name}
        response = requests.post(config.data_service_host, files=files, data=data)
        if response.status_code == 200:
            log.debug("Image successfully sent to firm-fms")
        else:
            log.error("Error occured when trying to send image to firm-fms")

    return
