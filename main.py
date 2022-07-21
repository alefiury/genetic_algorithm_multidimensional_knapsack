import argparse

from omegaconf import OmegaConf

from utils.genetic_algorithm import GeneticAlgorithm
from utils.utils import split_metadata

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

    num_variables, num_constraints, opt, coeficients, constraints, max_weigths = split_metadata(cfg.data.metadata_path)

    ga = GeneticAlgorithm(
        **cfg.genetic_algorithm,
        num_variables=num_variables,
        num_constraints=num_constraints,
        coeficients=coeficients,
        constraints=constraints,
        max_weigths=max_weigths
    )

    ga.iterate()


if __name__ == '__main__':
    main()