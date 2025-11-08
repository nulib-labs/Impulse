import argparse
import pathlib
from fireworks.core.launchpad import Workflow
from loguru import logger
import ocr_fireworks
from cli_helpers import list_image_files_in_dir, get_fireworks_data


def submit_image_processing_job_subcommand(args):
    fw_data = get_fireworks_data()
    lp = fw_data.get("LaunchPad", {})
    logger.info("Submitting image processing job")
    logger.info(f"Detected input value of: {args.input}")
    logger.info(f"Detected primary_key value of: {args.primary_key}")

    x = list_image_files_in_dir(args.input)
    logger.info(x)
    logger.info(f"Type of x: {type(x)}")
    logger.info(f"Found {len(x)} image-type files in {args.input}")
    fw = ocr_fireworks.define_firework("name", {}, x)
    print(fw)
    wf = Workflow(
        [fw], metadata={"accession_number": args.primary_key}, name=args.primary_key
    )

    lp.add_wf(wf)
    return None


def submit_ocr_job_subcommand(args):
    logger.info("Submitting OCR job")
    return None


def submit_subcommand(args):
    logger.info("Starting `submit` subroutine.")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # submit command
    submit = subparsers.add_parser("submit")
    submit.set_defaults(func=submit_subcommand)

    jobtypes = submit.add_subparsers(dest="jobtype", required=True)

    # image_processing subcommand
    ip = jobtypes.add_parser("image_processing")
    ip.add_argument("--input", "-i", type=pathlib.Path)
    ip.add_argument("--primary_key", "-pk", type=str)
    ip.set_defaults(func=submit_image_processing_job_subcommand)

    submit.set_defaults(func=submit_image_processing_job_subcommand)
    args = parser.parse_args(
        "submit image_processing -i ./pilot_test_data/p1074_35556031825029/".split()
    )
    args.func(args)


if __name__ == "__main__":
    main()
