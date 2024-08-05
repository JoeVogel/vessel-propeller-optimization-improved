#!/usr/bin/python3
###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# argv[1] is the candidate configuration number
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the instance name
# The rest (argv[5:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import datetime
import os.path
import re
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from cma_single_run import run

## This a dummy example that shows how to parse the parameters defined in
## parameters.txt and does not need to call any other software.

# Useful function to print errors.
def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(
            "\nUsage: ./target-runner.py <configuration_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
        sys.exit(1)

    # Get the parameters as command line arguments.
    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = float(sys.argv[4])
    cand_params = sys.argv[5:]

    # Parse parameters
    while cand_params:
        # Get and remove first and second elements.
        param = cand_params.pop(0)
        value = cand_params.pop(0)
        if param == "--sigma":
            sigma = float(value)
        elif param == "--num_population":
            num_population = int(value)
        else:
            target_runner_error("unknown parameter %s" % (param))

    # Sanity checks
    if None in [sigma, num_population]:
        target_runner_error("Missing parameter value!")

    # Run and print the output
    try:
        fitness = run(instance, sigma, num_population)
        print(fitness)
        sys.exit(0)
    except Exception as e:
        target_runner_error(f"Exception occurred: {str(e)}")




