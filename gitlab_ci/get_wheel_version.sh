#!/usr/bin/env bash


get_wheel_version () {
    REGEX='^(.*)\/([a-zA-Z0-9_]+)-([a-zA-Z0-9.]+)(.*)-py(.+)\.whl$'

    wheel="$1"

    if [[ ${wheel} =~ ${REGEX} ]]; then
        location=${BASH_REMATCH[1]}
        package_name=${BASH_REMATCH[2]}
        package_version_global=${BASH_REMATCH[3]}
        package_version_local=${BASH_REMATCH[4]}
        package_build_info="py${BASH_REMATCH[5]}"
        echo "${package_version_global}${package_version_local}"
    else
        echo 'ERROR: Incorrect wheel'
        exit 1
    fi
}


get_project_name () {
    path=${1}
    export -f get_wheel_version
    find ${path} -type f -name '*.whl' -exec bash -c 'get_wheel_version "$0"' {} \;
}

get_project_name ${1}