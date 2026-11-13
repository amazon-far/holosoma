#!/usr/bin/env bash
# Released Isaac Sim script alias for Isaac Lab 2.
if [[ -n ${BASH_SOURCE[0]:-} ]]; then
    _isaaclab_alias_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
elif [[ -n ${ZSH_VERSION:-} ]]; then
    _isaaclab_alias_dir=$(cd -- "$(dirname -- "${(%):-%x}")" && pwd)
fi
source "$_isaaclab_alias_dir/source_isaaclab2_setup.sh" && unset _isaaclab_alias_dir
