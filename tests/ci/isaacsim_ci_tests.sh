#!/usr/bin/env bash
# Released test-script alias for Isaac Lab 2.
if [[ $# == 0 ]]; then set -- --backend isaaclab2; fi
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/isaaclab_ci_tests.sh" "$@"
