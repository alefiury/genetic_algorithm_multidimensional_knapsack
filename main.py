import argparse
import logging

import numpy as np
from omegaconf import OmegaConf

from utils.genetic_algorithm import GeneticAlgorithm
from utils.utils import split_metadata, write_results, formatter_single

# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        default='config/default.yaml',
        type=str,
        help="YAML file with configurations"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    problems = split_metadata(cfg.data.metadata_path)

    for idx, problem in enumerate(problems):
        log.info(f"Problem: {idx+1}/{len(problems)}")
        results = []
        for run in range(cfg.execution.num_runs):
            log.info(f"Run: {run+1}/{cfg.execution.num_runs}")
            ga = GeneticAlgorithm(
                **cfg.genetic_algorithm,
                **problem
            )

            best_fitness, avg_fitness, best_individual = ga.iterate()

            results.append([best_fitness, best_individual])

        best_fitness_list = np.array(results)[:, 0].astype(float)
        mean_best_fitness = np.mean(best_fitness_list)
        best_idx = np.argmax(best_fitness_list)

        write_results(
            problem_name=str(problem["num_constraints"]) + '.' + str(problem["num_variables"]) + '-' + str(idx),
            best_fitness=results[best_idx][0],
            mean_best_fitness=mean_best_fitness,
            best_individual=results[best_idx][1],
            output_file_path=cfg.log.output_path
        )

if __name__ == '__main__':
    main()