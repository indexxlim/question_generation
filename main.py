'''
    Question Generation Training
'''

import datetime
import os
import re
import logging
import argparse
import torch

from configloader import train_config
import transformers
from dataloader import QGDataset, QGBatchGenerator, get_dataloader
from train import train


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"  # Set the GPUs 


def gen_checkpoint_id(args):
    project_id = args.PROJECT_ID
    timez = datetime.datetime.now().strftime("%Y%m%d%H%M")
    checkpoint_id = "_".join([project_id, timez])
    return checkpoint_id

def get_logger(args):
    log_path = f"{args.checkpoint}/info"

    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    train_instance_log_files = os.listdir(log_path)
    train_instance_count = len(train_instance_log_files)


    logging.basicConfig(
        filename=f'{args.checkpoint}/info/train_instance_{train_instance_count}_info.log',
        filemode='w',
        format="%(asctime)s | %(filename)15s | %(levelname)7s | %(funcName)10s | %(message)s",
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.info("-"*40)
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("-"*40)

    return logger

def checkpoint_count(checkpoint):
    _, folders, files = next(iter(os.walk(checkpoint)))
    files = list(filter(lambda x :"saved_checkpoint_" in x, files))

    checkpoints = map(lambda x: int(re.search(r"[0-9]{1,}", x).group()[0]),files)

    try:
        last_checkpoint = sorted(checkpoints)[-1]
    except:
        last_checkpoint = 0
    return last_checkpoint

def get_args():
    global train_config

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--PROJECT_ID",
        type=str,
        default=train_config.tbai_required.project_id
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default=train_config.setup.model_class
    )
    parser.add_argument(
        "--tokenizer_class",
        type=str,
        default=train_config.setup.tokenizer_class
    )
    parser.add_argument(
        "--optimizer_class",
        type=str,
        default=train_config.setup.optimizer_class
    )
    parser.add_argument(
        "--model",
        type=str,
        default=train_config.setup.model
    )    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=train_config.setup.tokenizer
    )
    parser.add_argument(
        "--device",
        type=str,
        default=train_config.setup.device
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=train_config.setup.checkpoint if hasattr(train_config.setup, "checkpoint") else None
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=train_config.setup.data_dir
    )
    parser.add_argument(
        "--distributed",
        type=bool,
        default=train_config.setup.distributed
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=train_config.hyperparameters.learning_rate
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_config.hyperparameters.epochs
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=train_config.hyperparameters.train_batch_size
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=train_config.hyperparameters.eval_batch_size
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=train_config.hyperparameters.gradient_accumulation_steps
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=train_config.hyperparameters.log_every
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        default=train_config.hyperparameters.fp16
    )
    parser.add_argument("--local_rank", default=0, type=int)

    args=parser.parse_args()
    args.device = args.device if args.device else 'cpu'

    return args


def main():
    # Get ArgParse
    args=get_args()
    print(args)

    if args.checkpoint:
        args.checkpoint = (
            "./model_checkpoint/" + args.checkpoint[-1]
            if args.checkpoint[-1] == "/"
            else "./model_checkpoint/" + args.checkpoint
        )
    else:
        args.checkpoint = "./model_checkpoint/" + gen_checkpoint_id(args)
    
    # Define Model
    model_class = getattr(transformers, args.model_class)

    # if os.path.isdir(args.checkpoint):
    #     args.checkpoint_count = checkpoint_count(args.checkpoint)
    #     logger = get_logger(args)
    #     logger.info(f"Checkpoint path directory exists")
    #     logger.info(f"Loading model from saved_checkpoint_{args.checkpoint_count}")
    #     model = torch.load(f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")

    #     args.checkpoint_count +=1

    # else:
    try:
        os.makedirs(args.checkpoint)
    except:
        print("Ignoring Existing File Path ...")

    model = model_class.from_pretrained(args.model)
    model.model_parallel = args.distributed
    if args.distributed:
        model.parallelize()
        #model.cuda()
    #else:
        #model.to(args.device)



    args.checkpoint_count = 0
    logger = get_logger(args)

    logger.info(f"Model Creation")
    logger.info(f"Checkpoint creation {args.checkpoint}")

    args.logger = logger
    
    
    tokenizer_class = getattr(transformers, args.tokenizer_class)
    tokenizer = tokenizer_class.from_pretrained(args.model)
    
    optimizer_class = getattr(transformers, args.optimizer_class)
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)
    
    logger.info(f"Loading data from {args.data_dir} ...")
    
    train_dataset = QGDataset(data_path=os.path.join(args.data_dir, "MRC_Text_train_data_300K_2021-08-09.json"))
    test_dataset = QGDataset(data_path=os.path.join(args.data_dir, "MRC_Text_test_data_300K_2021-08-09.json"))
    
    batch_generator = QGBatchGenerator(tokenizer)
    
    train_loader = get_dataloader(
        train_dataset, 
        batch_generator=batch_generator,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
                             
    test_loader = get_dataloader(
        test_dataset, 
        batch_generator=batch_generator,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    
    try:
        # use wandb if possible: https://wandb.ai
        # first attempt: in the commandline, type: 
        # # wandb login 
        import wandb
        args.project_id = os.path.basename(args.checkpoint)
        wandb.init(project=args.project_id)
        wandb.config.update(args)
        wandb.watch(model, log="all")
    except:
        pass
    train(model, optimizer, tokenizer, train_loader, test_loader, args)

if __name__ == "__main__":
    main()