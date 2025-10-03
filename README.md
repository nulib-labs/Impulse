# Impulse

Impulse is a high-performance computing workflow manager designed primarily for running Deep Learning Document Models on Digitized Documents for Libraries and Cultural Heritage Institutions. It works on many image formats.

General architecture

File gets uploaded to MongoDB database. -> Supercomputing daemon triggers a compute job to remote supercomputing cluster -> remote worker returns generated text, images, etc. back into MongoDB

This allows for automation of downstream intake, such as Redivis, HathiTrust, Digital Collections, creation of IIIF manifests. Etc.
