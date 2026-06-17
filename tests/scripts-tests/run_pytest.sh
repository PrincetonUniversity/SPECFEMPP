#! /usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pytest "${SCRIPT_DIR}" -c "${SCRIPT_DIR}/pytest.toml"
