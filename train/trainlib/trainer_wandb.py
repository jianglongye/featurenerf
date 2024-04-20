import datetime
import os.path
import warnings

import numpy as np
import torch
import tqdm
import wandb


class TrainerWandb:
    def __init__(self, net, train_dataset, test_dataset, args, conf, device=None):
        self.args = args
        self.net = net
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.use_wandb = not args.no_wandb

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            persistent_workers=True,
        )
        self.test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=min(args.batch_size, 16),
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            persistent_workers=True,
        )

        self.num_total_batches = len(self.train_dataset)
        self.exp_name = args.name
        self.save_interval = conf.get_int("save_interval")
        self.print_interval = conf.get_int("print_interval")
        self.vis_interval = conf.get_int("vis_interval")
        self.eval_interval = conf.get_int("eval_interval")
        self.num_epoch_repeats = conf.get_int("num_epoch_repeats", 1)
        self.num_epochs = args.epochs
        self.accu_grad = conf.get_int("accu_grad", 1)
        date = datetime.datetime.now().strftime("_%m_%d")
        if self.use_wandb:
            wandb.init(project="feat_nerf", name=self.exp_name + date, config=conf)

        self.fixed_test = hasattr(args, "fixed_test") and args.fixed_test

        # Currently only Adam supported
        self.optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        if args.gamma != 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optim, gamma=args.gamma)
        else:
            self.lr_scheduler = None

        # Load weights
        self.managed_weight_saving = hasattr(net, "load_weights")
        if self.managed_weight_saving:
            net.load_weights(self.args)
        self.iter_state_path = "%s/%s/_iter" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.optim_state_path = "%s/%s/_optim" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.lrsched_state_path = "%s/%s/_lrsched" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.default_net_state_path = "%s/%s/net" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.start_iter_id = 0
        if args.resume:
            if os.path.exists(self.optim_state_path):
                try:
                    self.optim.load_state_dict(torch.load(self.optim_state_path, map_location=device))
                except:
                    warnings.warn("Failed to load optimizer state at", self.optim_state_path)
            if self.lr_scheduler is not None and os.path.exists(self.lrsched_state_path):
                self.lr_scheduler.load_state_dict(torch.load(self.lrsched_state_path, map_location=device))
            if os.path.exists(self.iter_state_path):
                self.start_iter_id = torch.load(self.iter_state_path, map_location=device)["iter"]
            if not self.managed_weight_saving and os.path.exists(self.default_net_state_path):
                net.load_state_dict(torch.load(self.default_net_state_path, map_location=device))

        self.conf = conf

    def post_batch(self, epoch, batch):
        """
        Ran after each batch
        """
        pass

    def extra_save_state(self):
        """
        Ran at each save step for saving extra state
        """
        pass

    def train_step(self, data, global_step):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, data, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()

    def vis_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None

    def start(self):
        def fmt_loss_str(losses):
            return "loss " + (" ".join(k + ":" + f"{losses[k]:5f}" for k in losses))

        def data_loop(dl):
            """
            Loop an iterable infinitely
            """
            while True:
                for x in iter(dl):
                    yield x

        test_data_iter = data_loop(self.test_data_loader)

        step_id = self.start_iter_id

        progress = tqdm.tqdm(bar_format="[{rate_fmt}] ")
        for epoch in range(self.num_epochs):
            if self.use_wandb:
                wandb.log({"lr": self.optim.param_groups[0]["lr"]}, step=step_id)

            batch = 0
            for _ in range(self.num_epoch_repeats):
                for data in self.train_data_loader:
                    losses = self.train_step(data, global_step=step_id)
                    loss_str = fmt_loss_str(losses)
                    if batch % self.print_interval == 0:
                        print(
                            "E",
                            epoch,
                            "B",
                            batch,
                            loss_str,
                            " lr",
                            self.optim.param_groups[0]["lr"],
                        )

                    if batch % self.eval_interval == 0:
                        test_data = next(test_data_iter)
                        self.net.eval()
                        with torch.no_grad():
                            test_losses = self.eval_step(test_data, global_step=step_id)
                        self.net.train()
                        test_loss_str = fmt_loss_str(test_losses)
                        print("*** Eval:", "E", epoch, "B", batch, test_loss_str, " lr")
                        if self.use_wandb:
                            wandb.log({"train/" + k: v for k, v in losses.items()}, step=step_id)
                            wandb.log({"test/" + k: v for k, v in test_losses.items()}, step=step_id)

                    if batch % self.save_interval == 0 and (epoch > 0 or batch > 0):
                        print("saving")
                        if self.managed_weight_saving:
                            self.net.save_weights(self.args)
                        else:
                            torch.save(self.net.state_dict(), self.default_net_state_path)
                        torch.save(self.optim.state_dict(), self.optim_state_path)
                        if self.lr_scheduler is not None:
                            torch.save(self.lr_scheduler.state_dict(), self.lrsched_state_path)
                        torch.save({"iter": step_id + 1}, self.iter_state_path)
                        self.extra_save_state()

                    if batch % self.vis_interval == 0 and (epoch > 0 or batch > 0):
                        print("generating visualization")
                        if self.fixed_test:
                            test_data = next(iter(self.test_data_loader))
                        else:
                            test_data = next(test_data_iter)
                        self.net.eval()
                        with torch.no_grad():
                            vis, vis_vals = self.vis_step(test_data, global_step=step_id)
                        if self.use_wandb and vis_vals is not None:
                            wandb.log({"vis/" + k: v for k, v in vis_vals.items()}, step=step_id)

                        self.net.train()
                        if self.use_wandb and vis is not None:
                            vis_u8 = (vis * 255).astype(np.uint8)
                            wandb.log({"visual": wandb.Image(vis_u8)}, step=step_id)

                    if batch == self.num_total_batches - 1 or batch % self.accu_grad == self.accu_grad - 1:
                        # check if gradient is nan
                        for name, param in self.net.named_parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                print(f"detect {'nan' if torch.isnan(param).any() else 'inf'} in {name}")
                            if param.grad is not None and (
                                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                            ):
                                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                                print(f"detect {'nan' if torch.isnan(param.grad).any() else 'inf'} in {name}.grad")
                        self.optim.step()
                        # check if the model has nan parameters
                        for name, param in self.net.named_parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                print(f"detect {'nan' if torch.isnan(param).any() else 'inf'} in {name}")
                                breakpoint()
                        self.optim.zero_grad()

                    self.post_batch(epoch, batch)
                    step_id += 1
                    batch += 1
                    progress.update(1)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
