import os

import torch


class Checkpointer(object):

    def __init__(self, cfg, model, model_ema=None, optimizer=None, save_dir="", save_to_disk=None, logger=None, is_train=True):
        self.cfg = cfg
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.logger = logger
        self.is_train = is_train

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.model_ema is not None:
            data["model_ema"] = self.model_ema.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

        self.tag_last_checkpoint(save_file)

    def load(self, f=None, with_optim=True, load_mapping={}):
        if self.has_checkpoint() and self.is_train:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from ImageNet")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self.load_file(f)

        self.load_model(checkpoint)

        if with_optim:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def load_file(self, f):
        # load native pytorch checkpoint
        loaded = torch.load(f, map_location=torch.device("cpu"))

        return loaded

    def _load_model(self, checkpoint):
        if self.is_train and self.has_checkpoint():  # resume training
            self.model.load_state_dict(checkpoint["model"])
