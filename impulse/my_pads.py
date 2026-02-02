# Edited by Aerith Netzer,
from fireworks.core.launchpad import LaunchPad
from fireworks.utilities.filepad import FilePad
import os
import certifi
conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))


def get_mongo_client_kwargs():
    return {
        "tls": True,
        "tlsCAFile": certifi.where(),
    }


lp = LaunchPad(
    host=conn_str,
    port=27017,
    uri_mode=True,
    name="fireworks",
    mongoclient_kwargs=get_mongo_client_kwargs(),
)

fp = FilePad(
    host=conn_str,
    port=27017,
    name="fireworks",
    uri_mode=True,
    mongoclient_kwargs=get_mongo_client_kwargs(),
)
