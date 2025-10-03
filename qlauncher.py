from fireworks import LaunchPad, FWorker
import fireworks
from fireworks.queue.queue_launcher import launch_rocket_to_queue, rapidfire
import certifi
import os
from fireworks.utilities.fw_serializers import load_object_from_file


def main():
    # Build LaunchPad programmatically
    launchpad = fireworks.LaunchPad(
        host=os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"),
        port=27017,
        uri_mode=True,
        name="fireworks",
        mongoclient_kwargs={
            "tls": True,
            "tlsCAFile": certifi.where(),
        },
    )

    # Load QueueAdapter (your SLURM settings)
    qadapter = load_object_from_file(
        "/gpfs/projects/p32234/projects/aerith/Impulse/my_qadapter.yaml"
    )

    # Load FWorker definition
    fworker = FWorker.from_file(
        "/gpfs/projects/p32234/projects/aerith/Impulse/my_fworker.yaml"
    )

    launch_rocket_to_queue(launchpad, fworker, qadapter)
    # === Option B: loop until queue is empty (like `qlaunch rapidfire`) ===
    # rapidfire(launchpad, fworker, qadapter=qadapter, nlaunches=0)


if __name__ == "__main__":
    main()
