# Purpose:
This vectorstore is used to manage a chroma_db vectorstore and serve the vectorstore through http access.

I want a fast api in a docker container to host a chroma_db vectorstore. I need to be able to connect to the vectorstore via an http client. I need methods for managing collections, adding data, updating data, and deleting data and query and get as well. I need to be able to persist the chroma_db in the s3 bucket because my docker containers are not peristent volumes. I need to be able to backup the chroma_db vectorstore to the s3_bucket and download and use the chroma_db vectorstore from the s3_bucket. I need to be able to trigger this from an api endpoint. 

I need to be able to perform create, read, update, and delete operations for the training documents for the chroma_db vectorstore. 

Here is the documentation for a chroma db vectorstore: 

docs.trychroma.com/docs/run-chroma/client-server

## Directory Structure

.
├── .github
│   └── workflows
│       └── deploy.yml # This is used in the github actions to build the docker image and save it in docker hub. This github actions will then use the docker hub image in google cloud run and run the container in google cloud run.
├── .gitignore
├── app
│   ├── api # This holds api routes. Each route is included in main.py
│   │   └── __init__.py
│   ├── classes # These hold class definitions
│   │   └── __init__.py 
│   ├── core # This holds settings and configuration options
│   │   └── __init__.py
│   ├── db # This holds the database instance and class
│   │   ├── __init__.py
│   │   └── schema # this holds pydantic models
│   │       └── __init__.py
│   ├── __init__.py
│   ├── models # This holds huggingface models
│   │   └── __init__.py
│   └── service # This holds utility functions and function definitions used in the other files
│       └── __init__.py
├── docker-compose.yml # This is used to hot-reload and local dev testing (Dockerfile.dev)
├── Dockerfile.dev # This is used for the local dev testing image
├── Dockerfile # This is used for the production image in the google deploy script
├── prompt.md
├── README.md
└── requirements.txt

## S3_persistence Structure:
This is the S3 Persistence Structure
users/{user_id}/
├── vectorstore/                   # User-level vectorstore (chroma_db)
├── avatars/{avatar_id}/
│   ├── vectorstore_data/          # Avatar-specific context data (preprocessed documents for the chroma_db vectorstore)
│   ├── adapters/                  # QLoRA adapter files (the actual Adapter is stored here)
│   ├── adapters/training_data/    # Training data for fine-tuning (preprocessed for the LoRA Adapter)
|   └── media/                     # Unprocessed media for a specific avatar (audio/video/images/documents)
|── image/                         # User-level personal image
|── *{other_potential_user_level_folders}  # Other potential user-level folders such as billing & account information

