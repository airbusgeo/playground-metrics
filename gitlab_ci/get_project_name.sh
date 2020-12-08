#!/usr/bin/env bash

get_project_name () {
    REGEX='^[ ]*name=['\''"]([A-Za-z0-9_-]+)['\''"],$'

    path=${1}

    CWD=$(pwd)

    cd ${path}
    line=$(grep '^[ ]*name=.*' setup.py)

    if [[ ${line} =~ ${REGEX} ]]; then
        cd ${CWD}
        echo ${BASH_REMATCH[1]}
    else
        cd ${CWD}
        exit 1
    fi
}

get_project_name ${1}
