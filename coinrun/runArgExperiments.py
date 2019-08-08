import subprocess

def main():

    timesteps = int(32e6)
    for architecture in ["impalalarge"]:
        for worker_count in ["16"]:
            for population_size in ["64"]:
                for num_envs in ["32"]:
                    for timesteps_agent in ["500"]:
                        for passthrough_perc in ["0.0625", "0.25"]:
                            for mutating_perc in ["0.0625", "0.25"]:
                                for mutation_rate in ["0.005", "0.01", "0.02"]:
                                    run_id = architecture + "-work" + worker_count + "-pop" + population_size + "-ne" + num_envs + "-ta" + timesteps_agent + "-passp" + passthrough_perc + "-mutp" + mutating_perc + "-mutr" + mutation_rate
                                    subprocess.call(["python", "-u", "-m", "coinrun.evolve_agent",
                                                    "--run-id", run_id,
                                                    "--architecture", architecture,
                                                    "--timesteps", str(timesteps),
                                                    "--worker-count", worker_count,
                                                    "--population-size", population_size,
                                                    "--num-envs", num_envs,
                                                    "--timesteps-agent", timesteps_agent,
                                                    "--passthrough-perc", passthrough_perc,
                                                    "--mutating-perc", mutating_perc,
                                                    "--mutation-rate", mutation_rate])

if __name__ == '__main__':
    main()