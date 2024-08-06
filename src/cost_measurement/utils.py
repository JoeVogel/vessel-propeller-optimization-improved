import psutil
import time

def measure_resources(algorithm, *args):
    process = psutil.Process()
    resources = []
    iteration = 1

    def measure_iteration():
        nonlocal iteration
        start_time = time.time()
        cpu_start = process.cpu_percent(interval=None)
        mem_start = process.memory_info().rss

        # Execute the algorithm for one iteration
        result = algorithm(iteration, *args)
        iteration += 1

        cpu_end = process.cpu_percent(interval=None)
        mem_end = process.memory_info().rss
        end_time = time.time()

        execution_time = end_time - start_time
        cpu_usage = cpu_end - cpu_start
        mem_usage = mem_end - mem_start

        resources.append((execution_time, cpu_usage, mem_usage))
        return result

    return measure_iteration, resources