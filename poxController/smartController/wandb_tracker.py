import wandb

class WandBTracker():

    def __init__(self, wanb_project_name, run_name, config_dict):
        self.wb_logger = wandb.init(
            # Set the project where this run will be logged
            project=wanb_project_name,
            name=run_name,
            # Track hyperparameters and run metadata
            config=config_dict)

