# Hyperparameters Tuning with 'DVC'
This repository contains **MNist handwritten digit** solutions for Hyperparameter Tuning as a Tutorial.

# Requirements

    pip install dvc==3.48.3
    pip install dvclive

# Tutorial
## Get Started
Step1: clone git repository

    git clone git@github.com:Hamzeluie/dvc_mnist.git
Step2: initialize dvc

    dvc init
Step3: commit changes in github

    git status
    git commit -m "Initialize DVC"

## DVC Stage
Each project contains some pipelines. for example, data prepration, train, validation and etc.,
which we call this **"stage"**.in this tutorial we just have one stage called train, which is used to train our model.

To define a stage

    dvc stage add -n <name> -p <parameters> -d <depecndencies file or dir> -o <output files or dir> -M <metrics file path does not track> -m <metrics file to track> --plots <plots path> <command to run the file>

some parameters are optional like (-m, -M, --plots, -o, ...). read more details in [here](https://dvc.org/doc/command-reference/stage/add)

our dvc stage add for this repo

    dvc stage add -n train -p mnist.epochs,mnist.lr,mnist.momentum -d src/train.py -M results/metrics.json  python src/train.py

After running the above command, a [dvc.yaml](dvc.yaml) file will be built in root repo directory

## DVC Live
To track some parameters like loss or accuracy, we should log their entire process(train).you can see complete description in [here](https://dvc.org/doc/dvclive)
in this repo, we use dvclive in [train.py](src/train.py#L80)(line 80 - 88)
## DVC Queue
To define experiment. you need dvc.yaml, train.py and  params.yaml file
* dvc.yaml: information about stage
* train.py: script of process or train.(when you run with python src/train.py this start training)
* params.yaml: the parameters and hyperparameters of the model

**note**: both dvc.yaml and params.yaml have parameters, but the difference is, params.yaml has all parameters and dvc.yaml just has the parameters which we want to tune them, not all of them.

    dvc exp run -n <experiment name> --set-param <define parameters and assign values>

you can see compelete definition of **"exp"** in [here](https://dvc.org/doc/user-guide/experiment-management/running-experiments#the-experiments-queue)

in this project we want to tune "lr" and "momentum" so the script is

    dvc exp run --name exp --queue -S "mnist.lr=0.1,0.01,0.001,0.0001" -S "mnist.momentum=0.01,0.1,0.5,0.9"

by running the above command, combination of values and parameters will be applied. but still won't be run.

to run experiment, execute the following command 

    dvc exp run --run-all

this command runs the experiments one at a time,
and stors loss and accuracy.

you can see the status of the experiment by using

    dvc queue status

also you can watch the results of the experiment by running the following command

    dvc exp show

**note**: "dvc exp show" shows all the details of the experiments; but status just calls the name of the experiments, running status shows(faild or Success) , created and task 

after this, run experiments to choose the best parameters or add it to "workspace" by using:

    dvc exp apply <experiment name>
then push by

    dvc exp push <remote name>


## DVC Metrics
To show metrics of the experiment of the "workspace", run:

    dvc metrics show

after running multiple experiments, then compare the differences of the experiments, run:

    dvc metrics diff

[full definition of the metrics](https://dvc.org/doc/command-reference/metrics)
## DVC Plots
To plot experiments, run:

    dvc plots show

to plot differences of the experiments, run:

    dvc plots diff

[full definition of the plot](https://dvc.org/doc/command-reference/plots)
## DVC Show Experiment
To save the results of the "dvc exp show", run:

    dvc exp show > exp.txt

this command writes the outputs of "dvc exp show" in a file named "exp.txt", so if you have any challenges with reading the terminal details, you can read it from the text file.
