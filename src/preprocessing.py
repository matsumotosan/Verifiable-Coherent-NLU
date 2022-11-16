import json
from www.dataset.prepro import get_tiered_data, balance_labels
from www.dataset.featurize import add_bert_features_tiered, get_tensor_dataset_tiered
from collections import Counter
import numpy as np
from www.dataset.ann import att_to_num_classes, idx_to_att, att_default_values
from sklearn.metrics import accuracy_score, f1_score
from www.utils import print_dict


def data_setup():
    # Preprocessing

    # We can split the data into multiple json files later
    data_file = 'all_data/www.json'
    with open(data_file, 'r') as f:
        dataset = json.load(f)


    # Data Filtering and Sampling
    cloze_dataset = {p: [] for p in dataset}
    order_dataset = {p: [] for p in dataset}

    for p in dataset:
        for exid in dataset[p]:
            ex = dataset[p][exid]

            if ex['type'] == None:
                continue
            
            ex_plaus = dataset[p][str(ex['story_id'])]

            if ex['type'] == 'cloze':
                cloze_dataset[p].append(ex)
                cloze_dataset[p].append(ex_plaus) # For every implausible story, add a copy of its corresponding plausible story

            # Exclude augmented ordering examples from dev and test, since the breakpoints aren't always accurate in those
            elif ex['type'] == 'order' and not (p != 'train' and ex['aug']): 
                order_dataset[p].append(ex)
                order_dataset[p].append(ex_plaus)
    
    data_file = 'all_data/www_2s_new.json'
    with open(data_file, 'r') as f:
        cloze_dataset_2s, order_dataset_2s = json.load(f)  

    for p in cloze_dataset_2s:
        label_dist = Counter([ex['label'] for ex in cloze_dataset_2s[p]])
        print('Cloze label distribution (%s):' % p)
        print(label_dist.most_common())
    
    return cloze_dataset_2s, order_dataset_2s


def get_baseline(cloze_dataset_2s, tokenizer):
    tiered_dataset = cloze_dataset_2s

    seq_length = 16 # Max sequence length to pad to

    tiered_dataset = get_tiered_data(tiered_dataset)
    tiered_dataset = add_bert_features_tiered(tiered_dataset, tokenizer, seq_length, add_segment_ids=True)
    
    return tiered_dataset

def get_tensor_dataset(tiered_dataset):
    # TODO: What the heck is train_spans. 
    # It has default value of False so maybe it's not necessary?
    tiered_tensor_dataset = {}
    max_story_length = max([len(ex['stories'][0]['sentences']) for p in tiered_dataset for ex in tiered_dataset[p]])
    for p in tiered_dataset:
        tiered_tensor_dataset[p] = get_tensor_dataset_tiered(tiered_dataset[p], max_story_length, add_segment_ids=True)
    return tiered_tensor_dataset


def run_baseline(tiered_dataset):
    # TODO: What is this used for???

    # Have to add BERT input IDs and tensorize again
    num_runs = 10
    stories = []
    pred_stories = []
    conflicts = []
    pred_conflicts = []
    preconditions = []
    pred_preconditions = []
    effects = []
    pred_effects = []
    verifiability = []
    consistency = []
    for p in tiered_dataset:
        if p == 'train':
            continue
    metr_avg = {}
    print('starting %s...' % p)
    for r in range(num_runs):
        print('starting run %s...' % str(r))
        for ex in tiered_dataset[p]:
            verifiable = True
            consistent = True

        stories.append(ex['label'])
        pred_stories.append(np.random.randint(2))

        if stories[-1] != pred_stories[-1]:
            verifiable = False

        labels_ex_p = []
        preds_ex_p = []

        labels_ex_e = []
        preds_ex_e = []

        labels_ex_c = []
        preds_ex_c = []

        for si, story in enumerate(ex['stories']):
            labels_story_p = []
            preds_story_p = []

            labels_story_e = []
            preds_story_e = []      

            for ent_ann in story['entities']:
                entity = ent_ann['entity']

            if si == 1 - ex['label']:
                labels_ex_c.append(ent_ann['conflict_span_onehot'])
                pred = np.zeros(ent_ann['conflict_span_onehot'].shape)
                for cs in np.random.choice(len(pred), size=2, replace=False):
                    pred[cs] = 1
                preds_ex_c.append(pred)

            labels_ent = []
            preds_ent = []
            for s, sent_ann in enumerate(ent_ann['preconditions']):
                if s < len(story['sentences']):
                    if entity in story['sentences'][s]:

                        labels_ent.append(sent_ann)
                        sent_ann_pred = []
                        for i, l in enumerate(sent_ann):
                            pl = np.random.randint(att_to_num_classes[idx_to_att[i]])
                            if pl > 0 and pl != att_default_values[idx_to_att[i]]:
                                if pl != l:
                                    verifiable = False
                            sent_ann_pred.append(pl)
                        preds_ent.append(sent_ann_pred)

            labels_story_p.append(labels_ent)
            preds_story_p.append(preds_ent)

            labels_ent = []
            preds_ent = []
            for s, sent_ann in enumerate(ent_ann['effects']):
                if s < len(story['sentences']):
                    if entity in story['sentences'][s]:
                        labels_ent.append(sent_ann)
                        sent_ann_pred = []
                        for i, l in enumerate(sent_ann):
                            pl = np.random.randint(att_to_num_classes[idx_to_att[i]])
                            if pl > 0 and pl != att_default_values[idx_to_att[i]]:
                                if pl != l:
                                    verifiable = False
                            sent_ann_pred.append(pl)
                        preds_ent.append(sent_ann_pred)

                labels_story_e.append(labels_ent)
                preds_story_e.append(preds_ent)

            labels_ex_p.append(labels_story_p)
            preds_ex_p.append(preds_story_p)

            labels_ex_e.append(labels_story_e)
            preds_ex_e.append(preds_story_e)

        conflicts.append(labels_ex_c)
        pred_conflicts.append(preds_ex_c)

        preconditions.append(labels_ex_p)
        pred_preconditions.append(preds_ex_p)

        effects.append(labels_ex_e)
        pred_effects.append(preds_ex_e)

        p_confl = np.nonzero(np.sum(np.array(preds_ex_c), axis=0))[0]
        l_confl = np.nonzero(np.sum(np.array(labels_ex_c), axis=0))[0]
        assert len(l_confl) == 2, str(labels_ex_c)
        if not (p_confl[0] == l_confl[0] and p_confl[1] == l_confl[1]):
            verifiable = False    
            consistent = False

        verifiability.append(1 if verifiable else 0)
        consistency.append(1 if consistent else 0)

        # Compute metrics
        metr = {}
        metr['story_accuracy'] = accuracy_score(stories, pred_stories)

        conflicts_flat = [c for c_ex in conflicts for c_ent in c_ex for c in c_ent]
        pred_conflicts_flat = [c for c_ex in pred_conflicts for c_ent in c_ex for c in c_ent]
        metr['confl_f1'] = f1_score(conflicts_flat, pred_conflicts_flat, average='macro')

        preconditions_flat = [p for p_ex in preconditions for p_story in p_ex for p_sent in p_story for p_ent in p_sent for p in p_ent]
        pred_preconditions_flat = [p for p_ex in pred_preconditions for p_story in p_ex for p_sent in p_story for p_ent in p_sent for p in p_ent]
        metr['precondition_f1'] = f1_score(preconditions_flat, pred_preconditions_flat, average='macro')

        effects_flat = [p for p_ex in effects for p_story in p_ex for p_sent in p_story for p_ent in p_sent for p in p_ent]
        pred_effects_flat = [p for p_ex in pred_effects for p_story in p_ex for p_sent in p_story for p_ent in p_sent for p in p_ent]
        metr['effect_f1'] = f1_score(effects_flat, pred_effects_flat, average='macro')

        metr['verifiability'] = np.mean(verifiability)
        metr['consistency'] = np.mean(consistency)

        for k in metr:
            if k not in metr_avg:
                metr_avg[k] = []
            metr_avg[k].append(metr[k])

    for k in metr_avg:
        metr_avg[k] = (np.mean(metr_avg[k]), np.var(metr_avg[k]) ** 0.5)
    print('RANDOM BASELINE (%s, %s runs)' % (str(p), str(num_runs)))