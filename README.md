# CIFAR10 Project as MLOps Playground

Please have a look over https://dvc.org/doc/start to get a general understanding of the dvc tool, before diving into the code.

## Prefix

* DVC lets us define ML pipelines and thereby:
    * track and version input data
        * e.g. our image files
    * track and version output data
        * e.g. our model checkpoints, tensorboards, metrics and plots
    * flexibly define DAG structured pipelines, where each stage consists of:
        - a stage name
        - parameters (DL related parameters typically like learning_rate , max_epochs, validation split, ...)
        - dependencies (mostly related to some file/folder paths, e.g. some input or output folders as well as script files)
    * cache intermediate outputs, such that we only need to run the stages that were subject to change
* This repo is a small toy example that:
    * sets up the dvc repo automatically
    * tracks the input data
    * defines a pipeline, consisting of:
        * data preparation
        * model training
        * model evaluation
    * logs the outputs to mlflow (not yet using dvc, but also easily possible)
        * https://dvc.org/doc/command-reference/metrics
        * https://dvc.org/doc/command-reference/plots
        * trained models can be versioned similarly to how we version the input data
    * uses a gitlab-runner on dgx2 to run the pipeline in ci/cd fashion


## Setup
* If you want to reproduce the pipeline created in this repo, you need to:
    * **fork** the repo (cloning won't be enough)
    * run `dvc destroy` to get rid off the already setup dvc tracking in place
    * change the filepaths to your user (i.e. `/raid/<user>/..`). Affected files are:
        * dvc_prepare.sh
        * dvc_train.sh
        * dvc_evaluate.sh
        * run.sh
        * Dockerfile.setup
        * dvc_setup.sh
        * Dockerfile
        * run_setup.sh
    * change the s3 bucket defined in `dvc_setup.sh` for backup storage
* In order to setup the project, you need to:
    * setup dvc
    * download data
    * track the data
    * configuring the remote storage
* All of those steps are described in `dvc_setup.sh` that is run by the `Dockerfile.setup`
* In order to run the setup, run: ` ./build_setup.sh && ./run_setup.sh `


## Usage
* Simply commit & push changes
* MANUALLY start the gitlab runner on gitlab to run the pipeline
    * IMPORTANT: Do not make the ci/cd step automatic, as this will result in an infinite loop of runs as the pipeline runs `git push` itself
* The gitlab runner procedure is described in `.gitlab-ci.yml`

## Output
* The output of the run is logged on mlflow http://62.96.249.154:5040/#/experiments
* This repo logs:
    * simple config parameters
    * simple result metrics
    * plots as artifacts
    * the model as artifact

## Things to pay attention to
* Pipeline should only be run on master branch
* Development should be done on some other branch
    * and finished changes merged into the master branch
* Before another round of ci/cd can be launched on the master branch, the previous run needs to have finished first and you need to `git pull` the changes in the `dvc` log files
    * each ci/cd run will cause a change in the dvc log files, so in order not to corrupt those files we need to wait
    * hopefully this will become easier with the next release of dvc

## Things the toy repo does not cover at the moment
* The repo does not:
    * version output files, such as model weights or tensorboards
        * Larger output files such as tensorboards and model weights can be tracked similarly as we track the data
            * use the `dvc add` workflow on the `/raid` folder
    * do metric logging, as this is currently done in mlflow
        * choice was made due to nicer UI, but we can certainly also track that in dvc or have it in both
        * Outputs such as metrics and plots can be added into the workflow easily as shown in:
            * https://dvc.org/doc/command-reference/metrics
            * https://dvc.org/doc/command-reference/plots

## Open points:
* for proper projects, our data needs to be backed up to a proper s3 bucket
    * not the toy bucket I am using for this repo
* does the dgx2 runner scale when multiple projects run ci/cd simultaneously?
