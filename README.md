# CIFAR10 Project as MLOps Playground

## Structure
* The data and dvc cache is handled externally on `/raid/nicolas`
    * for this project it would have been fine in the repo workspace, but for proper projects, the data size will be too large

## Usage
* Running the first time:
    * run: `obtain_data.sh`
        * downloads data
        * starts tracking the data using dvc
    * run: `configure_remote_storage.sh`
        * have a remote backup of the data on s3
* Run: `./run_locally.sh`
    * builds the docker image
    * runs the dvc pipeline
        * preparing data
        * training
        * evaluation
            * logs the results to mlflow
            * logs the trained model as artifact on mlflow
    * pulls the config changes into local repo


## Open points:
* need to back up the data to proper s3 bucket (not just some toy bucket)
    * reconfigure remote storage
