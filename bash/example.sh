#!/usr/bin/env bash

# parse flags - market, simml, both
while getopts ":ab" opt; do
  case ${opt} in
    a ) echo "Proc 1" ;;
    b ) echo "Proc 2" ;;
  esac
done

# wait for p1 to get externals and build
./proc_01.sh &
./proc_02.sh &
wait
echo "Finished building all"

# start p1
echo "try start "
open -a Terminal.app proc_01.sh

# start up proc 2 in new window
open -a Terminal.app proc_02/proc_02.sh

function execute_proc_01() {
  ./proc_01.sh &
}
