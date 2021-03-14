import sys
sys.path.append('/home/foksly/Documents')

from shad.nn.trainer import DefaultTrainer
CONFIG_PATH = 'config.yaml'

if __name__ == "__main__":
    trainer = DefaultTrainer.configure(CONFIG_PATH)
    trainer.run()
