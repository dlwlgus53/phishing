import pdb

def evaluate_metrics(all_prediction, from_which, slot_temp = None):
    pdb.set_trace()
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for key in all_prediction.keys():
        for t in all_prediction[key].keys():
            
            turn_belief_set = list(set(all_prediction[key][t]))
            from_which_set = list(set(from_which[key][t]))
            # if set(cv["turn_belief"]) == set(cv[from_which]):
            if turn_belief_set == from_which_set:
                joint_acc += 1
            total += 1

            # Compute prediction slot accuracy
            # # temp_acc = compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
            # temp_acc = compute_acc(turn_belief_set, from_which_set, slot_temp)
            # turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(turn_belief_set, from_which_set)
            F1_pred += temp_f1
            F1_count += count

    joint_acc_score = joint_acc / float(total) if total!=0 else 0
    # turn_acc_score = turn_acc / float(total) if total!=0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
    print(joint_acc_score)
    print(F1_score)
    
    return joint_acc_score, F1_score #, turn_acc_score


def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        if len(pred)==0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count