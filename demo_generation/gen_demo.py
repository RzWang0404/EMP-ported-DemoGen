from omegaconf import OmegaConf
import hydra
import pathlib

from demo_generation.demogen import DemoGen


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('demo_generation', 'config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    generator: DemoGen = cls(cfg)
    generator.generate_demo()


if __name__ == "__main__":
    main()
    
"""from omegaconf import OmegaConf
import os
import numpy as np
import hydra
import pathlib

from demo_generation.demogen import DemoGen

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('demo_generation', 'config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    
    # Initialize DemoGenEMP
    demo_gen = DemoGen(cfg)
    
    # Enable orientation tracking if specified in config
    if cfg.get('use_orientation', False):
        demo_gen.emp_planner.enable_orientation_tracking(True)
    
    # Generate demos
    demo_gen.generate_demo()
    
    print("Demo generation complete!")

if __name__ == "__main__":
    main()
"""

