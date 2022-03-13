import json
import torch
import logging
import pdb
import argparse
from dataset import Dataset
import init
import ontology
from collections import defaultdict, OrderedDict
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor

parser = argparse.ArgumentParser()


parser.add_argument('--max_length' ,  type = int, default=128)
parser.add_argument('--data_rate' ,  type = float, default=0.01)
parser.add_argument('--do_train' ,  type = int, default=0)
parser.add_argument('--do_short' ,  type = int, default=0)
parser.add_argument('--dst_student_rate' ,  type = float, default=1.0)
parser.add_argument('--seed' ,  type = int, default=1)
parser.add_argument('--batch_size' , type = int, default=4)
parser.add_argument('--test_batch_size' , type = int, default=16)
parser.add_argument('--port' , type = int,  default = 12355)
parser.add_argument('--max_epoch' ,  type = int, default=1)
parser.add_argument('--base_trained', type = str, default = 'google/mt5-small', help =" pretrainned model from 🤗")
parser.add_argument('--base_pretrained' , type = str,  help = 'base_pretrainned model')
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--debugging' , type = bool,  default = False, help = "Don't save file")
parser.add_argument('--test_path' , type = str,  default = './data/web.json')
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
args.world_size = args.gpus * args.nodes 
args.tokenizer = T5Tokenizer.from_pretrained(args.base_trained)

class Model:
    def __init__(self, model_path):
        args.pretrained_model = model_path
        state_dict = torch.load(args.pretrained_model)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        self.dst = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to('cuda:0')
        self.dst.load_state_dict(new_state_dict)
    
    def inference(self, args, model, test_loader, test_dataset):
        belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id # schema
        model.eval()
        with torch.no_grad():
            for iter,batch in enumerate(test_loader):
                outputs_text = model.generate(input_ids=batch['input']['input_ids'].to('cuda'))
                outputs_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]
                
                for idx in range(len(outputs_text)):
                    dial_id = batch['dial_id'][idx]
                    turn_id = batch['turn_id'][idx]
                    schema = batch['schema'][idx]
                    if turn_id not in belief_state[dial_id].keys():
                        belief_state[dial_id][turn_id] = {}
                    if outputs_text[idx] == ontology.QA['NOT_MENTIONED'] : continue
        
                    belief_state[dial_id][turn_id][schema] = outputs_text[idx]
                    test_dataset.belief_state[dial_id][turn_id][schema] = outputs_text[idx]
                

                if (iter + 1) % 50 == 0:
                    logger.info('step : {}/{}'.format(
                    iter+1, 
                    str(len(test_loader)),
                    ))
        turns = max(list(belief_state['temp'].keys()))
        return  dict(belief_state['temp'][turns])
    
    
    
    def data_to_json(self, data):
        temp = {}
        dials = data.split("\n")
        for idx, turn in enumerate(dials):
            if idx%2 == 0:
                temp[int(idx/2)] = {}
                temp[int(idx/2)]['user'] = turn
            else:
                temp[int(idx/2)]['system'] = turn
            temp[int(idx/2)]["belief"] = {}
            
        dial = {'temp' : temp}
        
        with open(args.test_path,'w') as f:
            json.dump(dial,f, indent=4, ensure_ascii=False)
            
        
    def get_result(self, data):
        print("inferencing")
        self.data_to_json(data)
        test_dataset =Dataset(args, args.test_path, 'test')
        loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.test_batch_size, pin_memory=True,
            num_workers=0, shuffle=False, collate_fn=test_dataset.collate_fn)
        result =  self.inference(args, self.dst , loader, test_dataset)
        print(result)
        return result

