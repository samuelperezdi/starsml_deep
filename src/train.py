import torch
from torch.utils.data import DataLoader
import yaml
from argparse import Namespace, ArgumentParser
import os
import wandb
from data import prepare_data, CreateDataset
from model import AstroCLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

os.environ["WANDB_MODE"] = "offline"

parser = ArgumentParser()
parser.add_argument(
    "--experiment_name", "-exp", type=str, help="name of experiment to load"
)
parser.add_argument("--device", "-dev", type=str, help="device to run on")
parser.add_argument(
    "--root", "-r", type=str, help="logdir", default="../results"
)
parser.add_argument("--name", "-n", type=str, help="name of run", default=None)

if __name__ == "__main__":
    # root is the parent of the directory containing this script
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_args = parser.parse_args()

    result_dir = f"{script_args.root}/{script_args.experiment_name}"
    run_args_file = f"{result_dir}/args.yaml"
    print(run_args_file)
    with open(run_args_file, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args = Namespace(**args)
    print("Loaded args:", args, "\n")

    if args.WANDB == "true":
        tags = args.TAGS if hasattr(args, "TAGS") else []
        wandb_logger = WandbLogger(project=args.WANDB_PROJECT_NAME, tags=tags, name=script_args.experiment_name, log_model="all")
        wandb_logger.experiment.config.update(args)

    data = prepare_data(args)

    torch.manual_seed(args.SEED)

    X_train_xray = data.X_train_cat1
    X_train_optical = data.X_train_cat2
    Y_train = data.y_train

    X_ev_xray = data.X_ev_cat1
    X_ev_optical = data.X_ev_cat2
    Y_ev = data.y_ev

    # to torch dataset
    train_ds = CreateDataset(
        X_train_xray,
        X_train_optical,
        Y_train,
        columns_xray=args.FEAT_XRAY,
        columns_optical=args.FEAT_OPTICAL
    )

    eval_ds = CreateDataset(
        X_ev_xray,
        X_ev_optical,
        Y_ev,
        columns_xray=args.FEAT_XRAY,
        columns_optical=args.FEAT_OPTICAL
    )

    train_loader = DataLoader(train_ds, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(eval_ds, batch_size=args.BATCH_SIZE, drop_last=True)
    # assuming args is already defined
    # model and data preparation
    model = AstroCLR(X_train_xray.shape[1], X_train_optical.shape[1], args.EMB_DIM)

    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(result_dir, "ckpts"), save_top_k=3, monitor="val_loss")
    #early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)

    # training
    trainer = pl.Trainer(
        max_epochs=args.EPOCHS,
        logger=wandb_logger if args.WANDB == "true" else False,
        callbacks=[checkpoint_callback],
        log_every_n_steps=args.LOG_INTERVAL,
        accelerator=args.DEV
    )

    trainer.fit(model, train_loader, val_loader)

    # save final model
    trainer.save_checkpoint(os.path.join(result_dir, "final_model.ckpt"))