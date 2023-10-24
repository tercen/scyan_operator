#!/bin/bash

# CHECK Both venv and .venv paths. 
# TODO Allow user to supply a virtual env name
# TODO Raise error if multiple venv paths are available
WD=`pwd`


if [ -z $1 ]
then
    echo "No name to create virtual envirmonment was specified. Aborting."
    echo "CALL: source setup.sh .venv"
else
    VENV_PATH=${WD}/${1}

    echo "Creating virtual environment at ${VENV_PATH}"

    if ! command -v python3-venv &> /dev/null
    then
        echo "python3-venv could not be found"
        sudo apt-get update
        sudo apt-get install python3-venv

    fi


    activate () {
    . ${VENV_PATH}/bin/activate
    }

    python3 -m venv ${VENV_PATH}
    activate
    python3 -m pip install -r requirements.txt
fi




