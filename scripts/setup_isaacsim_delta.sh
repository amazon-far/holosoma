#!/usr/bin/env bash
# Released Isaac Sim script alias for Isaac Lab 2.
if [[ -n ${BASH_SOURCE[0]:-} ]]; then
    _isaaclab_alias_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
elif [[ -n ${ZSH_VERSION:-} ]]; then
    _isaaclab_alias_dir=$(cd -- "$(dirname -- "${(%):-%x}")" && pwd)
fi
source "$_isaaclab_alias_dir/setup_isaaclab2_delta.sh"
unset _isaaclab_alias_dir
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    isaacsim_delta_main "$@"
fi
