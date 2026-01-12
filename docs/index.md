# Impulse

Digitization will deliver unprocessed files, I will do binarization/all that jazz
Pass it back to Digitization, they will do limb, METSXML stuff, and then they pass it back and do OCR

Impulse is a high-performance computing workflow manager designed primarily for running Deep Learning Document Models on Digitized Documents for Libraries and Cultural Heritage Institutions. It works on many image formats.

General architecture

File gets uploaded to MongoDB database. -> Supercomputing daemon triggers a compute job to remote supercomputing cluster -> remote worker returns generated text, images, etc. back into MongoDB

This allows for automation of downstream intake, such as Redivis, HathiTrust, Digital Collections, creation of IIIF manifests. Etc.

## Getting Started

### Reading/Writing to OCR Database Programmatically

Get a mongodb user set up with Aerith. Backup: Aihan. Backup for backup: Anyone in IT I think.

Review the `main.py` file in the repo. This goes through the steps of setting up a workflow and uploading your images to the database.

You will see some success message like `Successfully added workflow...`

If you get this far, nice!

Now, you will need access to the Quest supercomputing center. Ask Aerith to fill out a form that gets you access to the center via SSH.

You will then want to SSH into Quest, and clone this repository again and build whatever `venv` you want.

Now, you can run something like `qlaunch singleshot` to add the workflow you made into the queue. If you think that the job will need a lot of time. (i.e., more than a few thousand pages), you can edit the `my_qadapter.yaml` file and edit the walltime of the job. If you think that your number of pages will exceed the 48-hour time limit, I would highly recommend simply chunking the images into chapters/sections/whatever makes sense for your data.

## What do we need to do with this?

Currently, the roadmap consists of first getting it done so that the entire workflow can be done from the command line with relatively little user intervention.

We also need to fix rotations/deskew. My code sucks for this.

Finally, we need to standardize our data model, or at least visualize it so that people do not put their brain into pretzels trying to get a foreign key relation.

This is truly a project where getting 90% of the way there took 10% of the time.

Get Checksum of files
