import json
import pandas as pd
from tqdm import tqdm
import argparse
import gzip
import emoji
import re
import html


def clean_text(raw_text):
    if isinstance(raw_text, list):
        new_raw_text=[]
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            new_raw_text.append(raw.strip())
        cleaned_text = ' '.join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
 
    if cleaned_text==".":
        cleaned_text = ""
    new_text = emoji.replace_emoji(cleaned_text, replace='')

    return new_text


def transform_to_jsonl(df, filepath, hislen=20):
    user_list = set()
    item_list = set()
    seqs = []
    with open(filepath, "w", encoding="utf-8") as f:
        for row in df.iterrows():
            row_dict = row[1].to_dict()
            user = row_dict["user_id"]
            target = row_dict["parent_asin"]
            inter_his = row_dict["history"].split(' ')
            user_list.add(user)
            item_list.add(target)
            for item in inter_his:
                item_list.add(item)
            line = {'user_id': user, 'target_id': target, 'inter_history': inter_his[-hislen:]}
            seqs.append(line)
            f.write(json.dumps(line) + "\n")
    print("Successful write to " + filepath)
    return user_list, item_list, seqs


def load_meta_items(file, item2id):
    items = {}

    with gzip.open(file, "r") as fp:
        for line in tqdm(fp, desc="Load metas"):
            data = json.loads(line)
            item = data["parent_asin"]

            if item in item2id.keys():
                id = item2id[item]
            else:
                continue

            title = clean_text(data["title"]).strip(".!?,;:`")
            description = data["description"]
            description = clean_text(description)

            features = data["features"]
            features = clean_text(features)

            categories = data["categories"]
            categories = clean_text(categories)

            brand = ""
            if 'Brand' in data['details'].keys():
                brand = data['details']['Brand']
                brand = clean_text(brand)
            
            meta = title + ' ' + brand + ' ' + categories + ' ' + features + ' ' + description

            items[id] = {"title": title, "description": description, "brand": brand,
                         "features": features, "categories": categories,
                         "meta": meta }
    sorted_items = dict(sorted(items.items(), key=lambda x: x[0]))
    return sorted_items


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Musical_Instruments')
    parser.add_argument('--his_len', type=int, default=20)
    args = parser.parse_args()

    dataset = args.dataset

    train_csv = pd.read_csv(f"./dataset/{dataset}/{dataset}.train.csv.gz", compression='gzip')
    valid_csv = pd.read_csv(f"./dataset/{dataset}/{dataset}.valid.csv.gz", compression='gzip')
    test_csv = pd.read_csv(f"./dataset/{dataset}/{dataset}.test.csv.gz", compression='gzip')

    n_action = train_csv.shape[0] + valid_csv.shape[0] + test_csv.shape[0]

    train_csv = train_csv[~train_csv["history"].isna()]
    valid_csv = valid_csv[~valid_csv["history"].isna()]
    test_csv = test_csv[~test_csv["history"].isna()]
    
    users_1, items_1, train_seqs = transform_to_jsonl(train_csv, f"./dataset/{dataset}/{dataset}.train.jsonl", hislen=args.his_len)
    users_2, items_2, valid_seqs = transform_to_jsonl(valid_csv, f"./dataset/{dataset}/{dataset}.valid.jsonl", hislen=args.his_len)
    users_3, items_3, test_seqs = transform_to_jsonl(test_csv, f"./dataset/{dataset}/{dataset}.test.jsonl", hislen=args.his_len)

    users, items = set(), set()
    users.update(users_1, users_2, users_3)
    items.update(items_1, items_2, items_3)

    n_user = len(users)
    n_item = len(items)

    print(f"Total number of users: {n_user}")
    print(f"Total number of items: {n_item}")
    print(f"Total number of actions: {n_action}")
    print(f"Sparsity: {(1- n_action / (n_user * n_item))*100:.3f}%")

    item2id = {item: idx+1 for idx, item in enumerate(items)}

    with open(f"./dataset/{dataset}/{dataset}.user.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(list(users)))
    with open(f"./dataset/{dataset}/{dataset}.item.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(list(items)))
    with open(f"./dataset/{dataset}/{dataset}.emb_map.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(item2id))
    
    meta_file_path = f'./dataset/{dataset}/meta_{dataset}.jsonl.gz'
    meta_items = load_meta_items(meta_file_path, item2id)

    with open(f"./dataset/{dataset}/{dataset}.meta.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(meta_items))

    print(f"{dataset} dataset processed successfully!")
