import re
import pdb
import json
import torch
import pickle
import ontology
from tqdm import tqdm
import logging
from log_conf import init_logger
from collections import defaultdict
import random
logger = logging.getLogger("my")



class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, data_type):
        random.seed(args.seed)
        self.data_type = data_type
        self.tokenizer = args.tokenizer
        self.dst_student_rate = args.dst_student_rate
        self.max_length = args.max_length
        self.aux = args.aux
        self.domain = args.domain
        
        
        self.belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id
        self.gold_belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id
        

        self.gold_context= defaultdict(lambda : defaultdict(str))# dial_id, # turn_id
        
        self.data_type = data_type
        f = open("./use_list.txt", 'r')
        self.use_list = [line.strip() for line in f.readlines()]
        raw_path = f'{data_path}'
        
        if args.do_short:
            raw_path = f'./data/0306data.json'
            
            
        logger.info(f"load {self.data_type} raw file {raw_path}")   
        raw_dataset = json.load(open(raw_path , "r"))
        turn_id, dial_id,  question, schema, answer, gold_belief_state, gold_context= self.seperate_data(raw_dataset)

        assert len(turn_id) == len(dial_id) == len(question)\
            == len(schema) == len(answer)
            
        self.answer = answer # for debugging
        self.target = self.encode(answer)
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema
        self.gold_belief_state = gold_belief_state
        self.gold_context = gold_context
        
        
            
            
            
    def encode(self, texts ,return_tensors="pt"):
        examples = []
        for i, text in enumerate(texts):
            # Truncate
            while True:
                tokenized = self.tokenizer.batch_encode_plus([text], padding=False, return_tensors=return_tensors) # TODO : special token
                if len(tokenized.input_ids[0])> self.max_length:
                    idx = [m.start() for m in re.finditer("\[user\]", text)]
                    try:
                        text = text[:idx[0]] + text[idx[1]:] # delete one turn
                    except IndexError as e:
                        useable = self.max_length- (idx[0] + len("[user]"))
                        second_part = text[idx[0] + len("[user]"):]
                        short_second_part = second_part[len(second_part) - useable +1:]
                        text = text[:idx[0]] + "[user] " + short_second_part 
                else:
                    break
                
            examples.append(tokenized)
        return examples

    def __len__(self):
        return len(self.dial_id)

    def seperate_data(self, dataset):
        gold_belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id
        gold_context= defaultdict(lambda : defaultdict(str))# dial_id, # turn_id
        
        question = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []
        
        for d_id in dataset.keys():
            dialogue = dataset[d_id]
            dialogue_text = ""
            turn_ids = dialogue.keys()
            for t_id in turn_ids:
                turn = dialogue[t_id]
                turn_domain = turn['domain']
                
                if self.domain != 'all' and turn_domain != self.domain : break
                dialogue_text += '[user] '
                dialogue_text += str(turn['user'])

                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    if self.domain != 'all' and turn_domain != self.domain : continue
                    q = ontology.QA[key]['description']
                    if key in turn['belief']: # 언급을 한 경우
                        a = turn['belief'][key]
                        if isinstance(a, list) : a= a[0] # in muptiple type, a == ['sunday',6]
                    else:a = ontology.QA['NOT_MENTIONED']
                    
                    schema.append(key)
                    answer.append(a)
                    question.append(q)
                    dial_id.append(d_id)
                    turn_id.append(int(t_id))
                        
                        
                # ###########changed part ###########################################
                if self.data_type == 'train' and self.aux == 1:
                    for key_idx, key in enumerate(ontology.QA['all-domain']): 
                        domain = key.split("-")[0]
                        slot = key.split("-")[1]
                        if self.domain != 'all' and domain != self.domain : continue
                        
                        q = "대화에 " + domain + " " +slot  + ontology.QA["general-question"] +  "?" 
                        c = dialogue_text
                        if key in turn['belief']: # 언급을 한 경우
                            a = '네'
                        else:
                            a = '언급 없음'

                        schema.append(key)
                        answer.append(a)
                        question.append(q)
                        dial_id.append(d_id)
                        turn_id.append(int(t_id))
                # ########################################################################     
                    
                gold_belief_state[d_id][int(t_id)] = turn['belief']
                gold_context[d_id][int(t_id)] = dialogue_text
                dialogue_text += '[system] '
                dialogue_text += turn['system']

        return turn_id, dial_id,  question, schema, answer, gold_belief_state, gold_context

    def __getitem__(self, index):
        dial_id = self.dial_id[index]
        turn_id = self.turn_id[index]
        schema = self.schema[index]
        question = self.question[index]
        gold_context = self.gold_context[index]
        gold_belief_state = self.gold_belief_state[index]
        
        
        target = {k:v.squeeze() for (k,v) in self.target[index].items()}
        
        return {"target": target,"turn_id" : turn_id,"question" : question, "gold_context" : gold_context,\
            "dial_id" : dial_id, "schema":schema,  "gold_belief_state" : gold_belief_state }
    


    
    def make_DB(self, belief_state, activate):
        pass
    
    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        context = self.context[index]
        belief_state = self.belief_state[index]
        """
        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        question = [x["question"] for x in batch]
        schema = [x["schema"] for x in batch]
        target_list = [x["target"] for x in batch]
        

        belief = [self.belief_state[d][t-1]for (d,t) in zip(dial_id, turn_id)] 
        history = [self.gold_context[d][t] for (d,t) in zip(dial_id, turn_id)]
        
        input_source = [f"question: {q} context: {c} belief: {b}" for (q,c,b) in  \
            zip(question, history, belief)]
        
        source = self.encode(input_source)
        source_list = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
            
        pad_source = self.tokenizer.pad(source_list,padding=True)
        pad_target = self.tokenizer.pad(target_list,padding=True)
        
        return {"input": pad_source, "target": pad_target,\
                 "schema":schema, "dial_id":dial_id, "turn_id":turn_id}
        

if __name__ == '__main__':
    import argparse
    init_logger(f'data_process.log')
    logger = logging.getLogger("my")

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_rate' ,  type = float, default=1.0)
    parser.add_argument('--do_short' ,  type = int, default=0)
    parser.add_argument('--dst_student_rate' ,  type = float, default=0.0)
    parser.add_argument('--seed' ,  type = float, default=1)
    parser.add_argument('--aux' ,  type = int, default=1)
    
    parser.add_argument('--max_length' ,  type = int, default=128)
    parser.add_argument('--domain', type=str, default = '대출')
    
    args = parser.parse_args()

    args.data_path = '../phishing_origin_and_data_processing/data/0401dev_data.json'
    from transformers import T5Tokenizer
    args.tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
    
    dataset = Dataset(args, args.data_path, 'train')
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    t = args.tokenizer
    for batch in loader:
        for i in range(16):
            print(t.decode(batch['input']['input_ids'][i]))
            print(t.decode(batch['target']['input_ids'][i]))
            print()
            
        pdb.set_trace()
    
    