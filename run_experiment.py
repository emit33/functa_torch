from utils.config import Config
from utils.training import latentModulatedTrainer


if __name__ == "__main__":
    # Obtain config
    config = Config.from_yaml()

    trainer = latentModulatedTrainer(config.model, config.training, config.paths)
    trainer.train()
