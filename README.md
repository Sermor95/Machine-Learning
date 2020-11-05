## Docker application for Feature Selection

Docker container with a python application to study how different feature selection methods works in a dataset.

This container uses a mongo database to store the results obtained.

## Usage

Download the repository and use the following command within it.

```
docker-compose up
```

## Initialize database

To have a main base of results and not have to wait a long time for the first run, you only need a few steps:

- Access to mongo_express under `http://localhost:8081`.
- Create a new collection inside "machine_learning" database called 'fs_config'
- Create a new document with the data from "fs_config". (Create all documents at the same time)
- Create a new collection inside "machine_learning" database called 'fs_result'. (Create all documents at the same time)
- Create a new document with the data from "fs_result"

**Warning!**: Do not import collection with the import option from mongo_express!

## Explore application

Once the database is initialized, the next step is to access the application at `http://localhost:5000`.




