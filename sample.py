import utils
import time
import torch
import logging
import argparse
import datetime
from dataset import Dataset
import init
from collections import OrderedDict
from trainer import valid, train, test
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor

parser = argparse.ArgumentParser()


parser.add_argument('--max_length' ,  type = int, default=128)
parser.add_argument('--data_rate' ,  type = float, default=0.01)
parser.add_argument('--do_train' ,  type = int, default=1)
parser.add_argument('--do_short' ,  type = int, default=1)
parser.add_argument('--dst_student_rate' ,  type = float, default=1.0)
parser.add_argument('--seed' ,  type = int, default=1)
parser.add_argument('--batch_size' , type = int, default=4)
parser.add_argument('--test_batch_size' , type = int, default=16)
parser.add_argument('--port' , type = int,  default = 12355)
parser.add_argument('--max_epoch' ,  type = int, default=1)
parser.add_argument('--base_trained', type = str, default = "google/t5-large-ssm-nq", help =" pretrainned model from 🤗")
parser.add_argument('--base_pretrained' , type = str,  help = 'base_pretrainned model')
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--debugging' , type = bool,  default = False, help = "Don't save file")
parser.add_argument('--dev_path' ,  type = str,  default = '../KLUE/dev_data.json')
parser.add_argument('--train_path' , type = str,  default = '../KLUE/train_data.json')
parser.add_argument('--test_path' , type = str,  default = '../KLUE/test_data.json')
parser.add_argument('--detail_log' , type = int,  default = 0)
parser.add_argument('--save_prefix', type = str, help = 'prefix for all savings', default = '')
parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=4, type=int,help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,help='ranking within the nodes')
parser.add_argument('--aux' ,  type = int, default=1)
parser.add_argument('--train_continue', type=int, default = 0)

args = parser.parse_args()
init.init_experiment(args)
logger = logging.getLogger("my")


def load_trained(args,model, optimizer = None):
    logger.info(f"User pretrained model{args.pretrained_model}")
    state_dict = torch.load(args.pretrained_model)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    if optimizer:
        opt_path = "./model/optimizer/" + args.pretrained_model[7:] #todo
        optimizer.load_state_dict(torch.load(opt_path))
    print("load safely")
    
         
def get_loader(dataset,batch_size):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    shuffle = False
    pin_memory = True
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=0, shuffle=shuffle, sampler=train_sampler,  collate_fn=dataset.collate_fn)
    return loader       
    
def evaluate():
    test_dataset =Dataset(args, args.test_path, 'test')
    
    
    loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, pin_memory=True,
        num_workers=0, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    if args.pretrained_model:
        logger.info(f"User pretrained model{args.pretrained_model}")
        state_dict = torch.load(args.pretrained_model)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to('cuda:0')
        model.load_state_dict(new_state_dict)
        test(args, model, loader, test_dataset)
    
def main():
    utils.makedirs("./data"); utils.makedirs("./logs"); utils.makedirs("./model/optimizer"); utils.makedirs("./out");
    logger.info(args)
    args.world_size = args.gpus * args.nodes 
    args.tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    evaluate()

if __name__ =="__main__":
    main()
    

