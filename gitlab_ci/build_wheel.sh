#!/usr/bin/env bash

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# -------------------------------------------------------------------------------------------------------------------- #
# Your wheel build script here :

build_wheel () {
    apt-get update
    apt-get install -y libspatialindex-dev
    python setup.py bdist_wheel -d build
}

# -------------------------------------------------------------------------------------------------------------------- #

git_ref_type() {
    [[ -n "$1" ]] || echo "ERROR: Missing ref name"

    if git show-ref -q --verify "refs/heads/$1" 2>/dev/null; then
        echo "branch"
    elif git show-ref -q --verify "refs/tags/$1" 2>/dev/null; then
        echo "tag"
    elif git show-ref -q --verify "refs/remote/$1" 2>/dev/null; then
        echo "remote"
    elif git rev-parse --verify "$1^{commit}" >/dev/null 2>&1; then
        echo "hash"
    else
        echo "unknown"
    fi
    return 0
}


make_developement_version () {
    REGEX='^(.*)\/([a-zA-Z0-9_]+)-([a-zA-Z0-9.]+)(.*)-py(.+)\.whl$'
    DATE=$(date +%Y%m%d%H%M%S)

    wheel="$1"

    if [[ ${wheel} =~ ${REGEX} ]]; then
        echo 'Correct wheel... Proceed'
    else
        echo 'ERROR: Incorrect wheel'
        exit 1
    fi

    location=${BASH_REMATCH[1]}
    package_name=${BASH_REMATCH[2]}
    package_version_global=${BASH_REMATCH[3]}
    package_version_local=${BASH_REMATCH[4]}
    package_build_info="py${BASH_REMATCH[5]}"

    mkdir "/tmp/wheel_temp"
    unzip ${wheel} -d "/tmp/wheel_temp"

    echo "
    Matches for $wheel:

    location: $location
    name: $package_name
    version global: $package_version_global
    version_local: $package_version_local
    extras: $package_build_info

    dist-info: $package_name-$package_version_global.dist-info
    METADATA: $package_name-$package_version_global.dist-info/METADATA
    RECORD: $package_name-$package_version_global.dist-info/RECORD
    "

    # Change METADATA
    sed -i "s/Name: .*/Name: ${package_name}_dev/g" "/tmp/wheel_temp/$package_name-$package_version_global.dist-info/METADATA"
    sed -i "s/^Version: .*/Version: $package_version_global.dev$DATE$package_version_local/g" "/tmp/wheel_temp/$package_name-$package_version_global.dist-info/METADATA"
    # Change RECORD
    sed -i "s/$package_name-$package_version_global.dist-info\/\(.*\)/${package_name}_dev-$package_version_global.dev$DATE.dist-info\/\1/g" "/tmp/wheel_temp/$package_name-$package_version_global.dist-info/RECORD"
    # Change dist-info
    mv "/tmp/wheel_temp/$package_name-$package_version_global.dist-info" "/tmp/wheel_temp/${package_name}_dev-$package_version_global.dev$DATE.dist-info"

    rm -f ${wheel}
    CWD=$(pwd)
    cd /tmp/wheel_temp/
    zip -r "/tmp/wheel_temp.zip" *
    cd ${CWD}
    mv "/tmp/wheel_temp.zip" ${wheel}
    rm -rf "/tmp/wheel_temp"
    rm -f "/tmp/wheel_temp.zip"

    # Change wheel name
    mv "$wheel" "$(echo "$wheel" | \
        sed -n "s/\(.*\/\)\([a-zA-Z0-9_]*\)-\([a-zA-Z0-9\.]*\)\(.*\)-py\(.*\)$/\1\2_dev-\3.dev$DATE\4-py\5/p")"
}


build_wheel_for_ref () {
    git_ref=${1}
    echo "

* ============================================================================================================ *

Building wheel for ref: $git_ref

"
    git checkout ${git_ref}
    git pull
    build_wheel
    rm -r build/bdist.*
    rm -r build/lib
    echo "Git ref type: $(git_ref_type ${git_ref})..."
    if [[ $(git_ref_type ${git_ref}) == "branch" ]]; then
        echo "Make devel wheel."
        export -f make_developement_version
        find . -type f -name '*.whl' -exec bash -c 'make_developement_version "$0"' {} \;
    else
        echo "Make standard wheel."
    fi
}

if [[ ${1} = "" ]]; then
    echo "ERROR: Missing git ref"
    exit 1
fi

build_wheel_for_ref ${1}
git checkout ${CURRENT_BRANCH}
