import hydra

import toy_grid_dag


@hydra.main(config_path="configs", config_name="main") # use hydra==1.1
def main(cfg):
    class ARGS:
        pass
    args = ARGS()
    for k, v in cfg.items():
        setattr(args, k, v)

    toy_grid_dag.main(args)

if __name__ == "__main__":
    main()