import os
import argparse
import logging

import wandb
import numpy as np
from omegaconf import OmegaConf

from utils.genetic_algorithm import GeneticAlgorithm
from utils.utils import split_metadata, write_results, init_output_file, formatter_single

wandb.login()

# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--main_config_path',
        default='config/main_config.yaml',
        type=str,
        help="YAML file with configurations"
    )
    args = parser.parse_args()

    main_cfg = OmegaConf.load(args.main_config_path)
    os.makedirs(main_cfg.log.log_base_dir, exist_ok=True)

    # wandb_config = {
    #     **main_cfg.genetic_algorithm,
    #     **main_cfg.execution,
    #     **main_cfg.log,
    #     **main_cfg.data
    # }

    if main_cfg.log.wandb_log:
        wandb_mode = None
    else:
        wandb_mode = "disabled"


    for config_path in main_cfg.executions.config_paths:
        cfg = OmegaConf.load(config_path)
        wandb_config = {
            **cfg.genetic_algorithm
        }

        for metadata_path in main_cfg.data.metadata_paths:
            instance_name = os.path.basename(metadata_path).split('.')[0]
            # init_output_file(output_file_path=cfg.log.output_path)

            problems = split_metadata(metadata_path)

            for idx, problem in enumerate(problems[:main_cfg.executions.num_problems]):

                log.info(f"Problem: {idx+1}/{len(problems)}")
                results = []
                for run in range(main_cfg.executions.num_runs):
                    run_wandb = wandb.init(
                        project=f"{instance_name}-multidimensional_knapsack_{idx}-selection_{cfg.genetic_algorithm.selection}-repair_{cfg.genetic_algorithm.repair}",
                        config=wandb_config,
                        dir="wandb",
                        reinit=True,
                        mode=wandb_mode
                    )

                    wandb.run.name = f"Run {run}"
                    wandb.run.save()

                    log.info(f"Run: {run+1}/{main_cfg.executions.num_runs}")
                    ga = GeneticAlgorithm(
                        **cfg.genetic_algorithm,
                        **problem
                    )

                    best_fitness, avg_fitness, best_individual = ga.iterate()

                    results.append([best_fitness, best_individual])

                    run_wandb.finish()

                best_fitness_list = np.array(results)[:, 0].astype(float)
                mean_best_fitness = np.mean(best_fitness_list)
                std_best_fitness = np.std(best_fitness_list)
                best_idx = np.argmax(best_fitness_list)

                write_results(
                    problem_name=str(problem["num_constraints"]) + '.' + str(problem["num_variables"]) + '-' + str(idx),
                    mean_best_fitness=mean_best_fitness,
                    std_best_fitness=std_best_fitness,
                    best_individual=results[best_idx][1],
                    best_fitness=results[best_idx][0],
                    output_file_path=os.path.join(main_cfg.log.log_base_dir, f"{instance_name}-multidimensional_knapsack_{idx}-selection_{cfg.genetic_algorithm.selection}-repair_{cfg.genetic_algorithm.repair}")
                )

if __name__ == '__main__':
    main()