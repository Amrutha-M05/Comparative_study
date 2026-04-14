from copy import deepcopy
from multiprocessing import freeze_support

from src.main import run_experiment, CONFIG

MODELS = [
    "mobilenet_v2",
    "efficientnet_b0",
    "resnet50",
    "densenet121",
    "vgg16",
    "inception_v3",
]

def main():
    for model_name in MODELS:
        config = deepcopy(CONFIG)
        config["model_name"] = model_name
        print(f"\nRunning experiment for: {model_name}")
        run_experiment(config)

if __name__ == "__main__":
    freeze_support()
    main()