import os
from fireworks import Firework, FWorker, LaunchPad, ScriptTask, TemplateWriterTask, FileTransferTask
from fireworks.core.rocket_launcher import launch_rocket
import boto3
from tasks.my_tasks import DocumentExtractionTask, ImageProcessingTask
from tasks.mets import METSXMLToHathiTrustManifestTask

# # set up the LaunchPad and reset it
launchpad: LaunchPad = LaunchPad(uri_mode=True, host=os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING_IMPULSE"), name="fireworks")

# # create the Firework consisting of multiple tasks
fw = Firework(METSXMLToHathiTrustManifestTask(), spec={'impulse_identifier':"P0491_35556036056489", "s3_xml_path": "s3://nu-impulse-production/P0491_35556036056489/mets.xml", "s3_yaml_path": "s3://nu-impulse-production/P0491_35556036056489/mets.yaml"})
fw_id = fw.fw_id
launchpad.add_wf(fw)
