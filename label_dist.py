def get_labelcount_num(examples):
    label_count = defaultdict(lambda : 0)
    for i in examples:
        label_count[i['label']] += 1
    return label_count

def get_examples(dataset_name, split, stem, n_shot, variant):
    if dataset_name == 'copa':
        from data_loaders import load_examples_copa
        examples = load_examples_copa(f'{stem}copa-{split}.xml')
        closed_label_space = False
        return get_labelcount_num(examples)

    elif dataset_name == 'copa-rev':
        from data_loaders import load_examples_copa_rev
        examples = load_examples_copa_rev(f'{stem}copa-{split}.xml')
        closed_label_space = False
    elif dataset_name == 'storycloze':
        from data_loaders import load_examples_storycloze
        examples = load_examples_storycloze(f'{stem}{split}.tsv')
        closed_label_space = False
        return get_labelcount_num(examples)

    elif dataset_name == 'hellaswag':
        from data_loaders import load_examples_hellaswag
        examples = load_examples_hellaswag(f'{stem}dev.jsonl')
        closed_label_space = False
        return get_labelcount_num(examples)

    elif dataset_name == 'race-m' or \
         dataset_name == 'race-h':
        from data_loaders import load_examples_race
        version = 'high' if dataset_name == 'race-h' else 'middle'
        examples = load_examples_race(stem, split, version)
        closed_label_space = False
        return get_labelcount_num(examples)

    elif dataset_name == 'arc-easy' or \
         dataset_name == 'arc-challenge':
        from data_loaders import load_examples_arc
        examples = load_examples_arc(f'{stem}{split}.jsonl')
        closed_label_space = False
        return get_labelcount_num(examples)
        
    elif dataset_name == 'obqa':
        from data_loaders import load_examples_obqa
        examples = load_examples_obqa(f'{stem}{split}.jsonl')
        closed_label_space = False
        return get_labelcount_num(examples)

    elif dataset_name == 'cqa':
        from data_loaders import load_examples_cqa
        if split == 'test':
            examples = load_examples_cqa(f'{stem}dev.jsonl')
        else:
            examples = load_examples_cqa(f'{stem}{split}.jsonl')
        closed_label_space = False
        return get_labelcount_num(examples)

    elif dataset_name == 'boolq':
        from data_loaders import load_examples_boolq
        examples = load_examples_boolq(f'{stem}dev.jsonl')
        closed_label_space = True
        label_count = defaultdict(lambda : 0)
        for i in examples:
            if i['label'] == 0:
                label_count['True'] += 1
            else:
                label_count['False'] += 1
        return label_count

    elif dataset_name == 'rte':
        from data_loaders import load_examples_rte
        examples = load_examples_rte(f'{stem}dev.jsonl')
        closed_label_space = True
        label_count = defaultdict(lambda : 0)
        for i in examples:
            if i['label'] == 0:
                label_count['True'] += 1
            else:
                label_count['False'] += 1
        return label_count

    elif dataset_name == 'cb':
        from data_loaders import load_examples_cb
        examples = load_examples_cb(f'{stem}dev.jsonl')
        closed_label_space = True
        label_count = defaultdict(lambda : 0)
        for i in examples:
            if i['label'] == 0:
                label_count['true'] += 1
            elif i['label'] == 1:
                label_count['false'] += 1
            else:
                label_count['neither'] += 1
        return label_count

    elif dataset_name == 'sst-2':
        from data_loaders import load_examples_sst2, load_examples_sst2_variants
        if n_shot > 0:
            examples = load_examples_sst2(f'{stem}{split}.tsv', f'{stem}/train.tsv', n_shot)
        elif variant is not None:
            examples = load_examples_sst2_variants(f'{stem}{split}.tsv', variant)
        else:
            examples = load_examples_sst2(f'{stem}{split}.tsv')
        closed_label_space = True
        label_count = defaultdict(lambda : 0)
        for i in examples:
            if i['label'] == 0:
                label_count['negative'] += 1
            else:
                label_count['positive'] += 1
        return label_count

    elif dataset_name == 'sst-5':
        from data_loaders import load_examples_sst5
        examples = load_examples_sst5(f'{stem}{split}.tsv')
        closed_label_space = True
        label_count = defaultdict(lambda : 0)
        for i in examples:
            if i['label'] == 0:
                label_count['very negative'] += 1
            elif i['label'] == 1:
                label_count['somewhat negative'] += 1
            elif i['label'] == 2:
                label_count['neutral'] += 1
            elif i['label'] == 3:
                label_count['somewhat positive'] += 1
            elif i['label'] == 4:
                label_count['very positive'] += 1
        return label_count

    elif dataset_name == 'agn':
        from data_loaders import load_examples_agn
        examples = load_examples_agn(f'{stem}{split}.csv')
        closed_label_space = True
        label_count = defaultdict(lambda : 0)
        for i in examples:
            if i['label'] == 0:
                label_count['World'] += 1
            elif i['label'] == 1:
                label_count['Sports'] += 1
            elif i['label'] == 2:
                label_count['Business'] += 1
            elif i['label'] == 3:
                label_count['Science'] += 1
        return label_count

    elif dataset_name == 'trec':
        split = 'train' if split == 'dev' else split
        from data_loaders import load_examples_trec
        examples = load_examples_trec(f'{stem}{split}.txt')
        closed_label_space = True
        label_count = defaultdict(lambda : 0)
        for i in examples:
            if i['label'] == 0:
                label_count['a description'] += 1
            elif i['label'] == 1:
                label_count['an entity'] += 1
            elif i['label'] == 2:
                label_count['a location'] += 1
            elif i['label'] == 3:
                label_count['a number'] += 1
            elif i['label'] == 4:
                label_count['an abbreviation'] += 1
            else:
                label_count['a person'] += 1
        return label_count

    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    return examples, closed_label_space

if __name__ == '__main__':
    from collections import defaultdict
    
    label_count_dataset = {"copa":{}, "storycloze":{}, "hellaswag":{}, "arc-easy":{}, "arc-challenge":{},
            "obqa":{}, "cqa":{}, "boolq":{}, "rte":{}, "cb":{}, "sst-2":{}, "sst-5":{}, "agn":{}, "trec":{}}
    for i in label_count_dataset.keys():
        label_count_dataset[i] = get_examples(i, "test", f'data/{i}/', 0, None)

    with open("label_count.json", 'w') as f:
        import json
        json.dump(label_count_dataset, f)