# Purpose:
This vectorstore is used to manage a chroma_db vectorstore and serve the vectorstore through http access.

I want a fast api in a docker container to host a chroma_db vectorstore. I need to be able to connect to the vectorstore via an http client. I need methods for managing collections, adding data, updating data, and deleting data and query and get as well. Here is the documentation for a chroma db vectorstore: 

docs.trychroma.com/docs/run-chroma/client-server