import torch
import pytorch_lightning as pl
import threading
import queue
import os
from torch_geometric.utils import to_dense_batch

class RepresentationTrackerCallback(pl.Callback):
    def __init__(self, frequency: str = "none", modules: dict = None):
        super().__init__()
        self.frequency = frequency
        self.modules_cfg = modules or {}
        self.save_queue = queue.Queue()
        self.hooks = []
        self.epoch = 0
        self.batch_idx = 0
        
        self.saver_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.saver_thread.start()

    def _save_worker(self):
        while True:
            item = self.save_queue.get()
            if item is None: break
            filepath, data = item
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(data, filepath)
            self.save_queue.task_done()
            
    def _get_hook(self, module_name):
        # We use with_kwargs=True if PyTorch >= 2.0, but to be safe for older versions,
        # we assume Probe receives (x, node_index) as positional args if passed that way,
        # or we just grab the output.
        def hook(module, inputs, output):
            x = output.detach().cpu()
            
            # If node_index was passed as the second positional argument to the probe:
            if len(inputs) > 1 and inputs[1] is not None:
                node_index = inputs[1].detach().cpu()
                
                # LIMIT TO FIRST 8 MOLECULES
                # node_index contains the graph index for each node.
                mask = node_index < 8
                x = x[mask]
                node_index = node_index[mask]
                
                data = {"x": x, "node_index": node_index}
            else:
                data = x
            
            # DETERMINING FILENAME
            if self.frequency == "val_first_and_last":
                # Save as 'first' if it's the very first epoch
                if self.epoch == 0:
                    fp = os.path.join(self.output_dir, "first", f"batch_{self.batch_idx}", f"{module_name}.pt")
                    self.save_queue.put((fp, data))
                
                # Always save to 'latest' (will override previous epochs)
                fp = os.path.join(self.output_dir, "latest", f"batch_{self.batch_idx}", f"{module_name}.pt")
                self.save_queue.put((fp, data))
            else:
                filepath = os.path.join(self.output_dir, f"epoch_{self.epoch:03d}", f"batch_{self.batch_idx}", f"{module_name}.pt")
                self.save_queue.put((filepath, data))
        return hook

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.epoch = trainer.current_epoch
        self.batch_idx = batch_idx
        log_dir = trainer.logger.save_dir if trainer.logger else "."
        name = trainer.logger.name if trainer.logger and trainer.logger.name else "run"
        self.output_dir = os.path.join(log_dir, name, "representations")
        
        if self.frequency == "val_first_batch" and batch_idx == 0:
            should_track = True
        elif self.frequency == "val_first_and_last" and batch_idx == 0:
            should_track = True
        elif self.frequency.startswith("val_every_") and "epochs" in self.frequency:
            n = int(self.frequency.split("_")[2])
            should_track = (self.epoch % n == 0) and (batch_idx == 0)
        else:
            should_track = False

        if should_track:
            for name, module in pl_module.named_modules():
                # E.g., name might be "model.parameterization.encoder.probes.post_sc_s"
                if any(k in name for k in self.modules_cfg):
                    # Clean the filename up based on the probe name
                    clean_name = name.split(".")[-1] 
                    h = module.register_forward_hook(self._get_hook(clean_name))
                    self.hooks.append(h)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def on_train_end(self, trainer, pl_module):
        self.save_queue.put(None)
        self.saver_thread.join()