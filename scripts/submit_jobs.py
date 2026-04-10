from fireworks import Firework, LaunchPad, Workflow
from fireworks.core.rocket_launcher import launch_rocket
from tasks.mets import METSXMLToHathiTrustManifestTask
from tasks.my_tasks import DocumentExtractionTask
from tasks.config import MONGO_URI
import boto3

launchpad: LaunchPad = LaunchPad(uri_mode=True, host=MONGO_URI, name="fireworks")


client = boto3.client('s3', region_name='us-west-2')
paginator = client.get_paginator('list_objects_v2')
operation_parameters = {'Bucket': 'nu-impulse-production',
                                        'Prefix': 'P0491_35556036056489/SOURCE/jpg/'}


page_iterator = paginator.paginate(**operation_parameters)
impulse_keys: list[str] = []
for page in page_iterator:
    for i in page['Contents']:
        print(i['Key'])
        impulse_keys.append(f"s3://nu-impulse-production/{i['Key']}")

ocr_fw: Firework = Firework(DocumentExtractionTask(), spec={'impulse_identifier': "P0491_35556036056489", "find_path_array_in": "keys", "keys": sorted(impulse_keys)}, name="Document Extraction Workflow")
launchpad.add_wf(ocr_fw)
