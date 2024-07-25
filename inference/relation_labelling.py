import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import treelib
import datasets
import copy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

def get_tree_dict(pickle_path):
    try:
        with open(pickle_path, 'rb') as file:
            tree = pickle.load(file)
            return tree
    except:
        print("Can't open file!")

def get_root_id(tree_dict):
    for key, item in tree_dict.items():
        if item['pnode_id'] == -1:
            root_id = key
    return root_id

def visualize_rst_tree(tree_dict, root_id, edu_list, new_relation=False, get_edu_text=False):
    rst_tree = treelib.Tree()
    relation_key = 'relation' if not new_relation else 'new_relation'
    node_list = [root_id]

    while node_list:
        id = node_list.pop()
        node = tree_dict[id]
        if (tree_dict.get(node['lnode_id']) is None) and (tree_dict.get(node['rnode_id']) is None):
            node_text = " EDU " + str(node['edu_span'])
            if get_edu_text:
                node_text += ": " + edu_list[node['edu_span'][0] - 1]
            rst_tree.create_node(node_text, id, parent=node['pnode_id'])
        else:
            node_text = node['node_form']

            if node['node_form'] == 'NN':
                node_text += "-" + tree_dict[node['rnode_id']][relation_key]
            elif node['node_form'] == 'NS':
                node_text += "-" + tree_dict[node['rnode_id']][relation_key]
            elif node['node_form'] == 'SN':
                node_text += "-" + tree_dict[node['lnode_id']][relation_key]
            else:
                raise ValueError("Unrecognized N-S form")

            if rst_tree.get_node(node['pnode_id']) is not None:
                rst_tree.create_node(node_text, id, parent=node['pnode_id'])
            else:
                rst_tree.create_node(node_text, id)
                print("\nNo parent at node: ", node_text, '\n')

        if tree_dict.get(node['rnode_id']) is not None:
            node_list.append(node['rnode_id'])
        if tree_dict.get(node['lnode_id']) is not None:
            node_list.append(node['lnode_id'])

    return rst_tree

def get_edus_from_file(edu_path):
    """Get EDUs from .edu file and return a list of EDUs
    """
    edus = []
    try:
        with open(edu_path, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                edus.append(line.rstrip('\n'))
        return edus
    except FileNotFoundError:
        print("Error: File not found!")

def extract_segments(edu_list, tree_dict, root_id): # use this when unpickling
    """Extract text segments from (1 or several EDUs) for relation labeller to read from (and make predictions)

    Args:
        edu_list: list containing EDUs (list)
        tree_dict: dict containing tree (dict)

    Return:
        Dict containing text of nucleus, satellite, and original relation from StageDP (dict)
    """

    segments = {'pnode_id': [], 'nucleus': [], 'satellite': [], 'original_relation': []} # if multi-nuclear, satellite represent second nucleus
    node_list = [root_id]
    while node_list:
        id = node_list.pop()
        node = tree_dict[id]

        if (tree_dict.get(node['lnode_id']) is None) and (tree_dict.get(node['rnode_id']) is None): # node is EDU
            continue

        left_edu_span = tree_dict[node['lnode_id']]['edu_span'] # tuple: (from, to)
        right_edu_span = tree_dict[node['rnode_id']]['edu_span'] # tuple: (from, to)

        # get corresponding text segments
        left_segment = ""
        for edu in range(left_edu_span[0], left_edu_span[1] + 1):
            left_segment += edu_list[edu - 1].strip() + ' '

        right_segment = ""
        for edu in range(right_edu_span[0], right_edu_span[1] + 1):
            right_segment += edu_list[edu - 1].strip() + ' '

        if node['node_form'] == 'NN':
            nucleus = left_segment
            satellite = right_segment
            relation = tree_dict[node['rnode_id']]['relation']
        elif node['node_form'] == 'NS':
            nucleus = left_segment
            satellite = right_segment
            relation = tree_dict[node['rnode_id']]['relation']
        elif node['node_form'] == 'SN':
            nucleus = right_segment
            satellite = left_segment
            relation = tree_dict[node['lnode_id']]['relation']

        segments['nucleus'].append(nucleus)
        segments['satellite'].append(satellite)
        segments['original_relation'].append(relation)
        segments['pnode_id'].append(id)

        if tree_dict.get(node['lnode_id']) is not None:
            node_list.append(node['lnode_id'])
        if tree_dict.get(node['rnode_id']) is not None:
            node_list.append(node['rnode_id'])

    return segments

def add_new_relations_to_tree_dict(tree_dict, new_relations):
    """Extract text segments from (1 or several EDUs) for relation labeller to read from (and make predictions)

    Args:
        tree_dict: dict containing tree (dict)
        new_relations: df containing parent id and new relations (and other components no considered in this method)

    Return:
        New modified tree_dict according to new relations identified
    """
    tree_dict_c = copy.deepcopy(tree_dict)
    for _, r in new_relations.iterrows():
        p_id = r['pnode_id']
        rel = r['new_relation']
        if tree_dict_c[p_id]['node_form'] == 'NN':
            tree_dict_c[tree_dict_c[p_id]['rnode_id']]['new_relation'] = rel
            tree_dict_c[tree_dict_c[p_id]['lnode_id']]['new_relation'] = rel
        elif tree_dict_c[p_id]['node_form'] == 'NS':
            tree_dict_c[tree_dict_c[p_id]['rnode_id']]['new_relation'] = rel
        elif tree_dict_c[p_id]['node_form'] == 'SN':
            tree_dict_c[tree_dict_c[p_id]['lnode_id']]['new_relation'] = rel

    return tree_dict_c

def write_to_text_file(text_path, text):
    """Write string in text to text_path. The text is to be analyzed using RST and generated questions from.

    Args:
        text_path (str): path of file to write to
        text (str): text to write to (informational text to extract questions from)

    """

    try:
        with open(text_path, 'w') as f:
            f.write(text)
    except:
        print("Can't open text file!\n")

def prep_data(df):
    # adding <sep> token between nucleus and satellite
    separation_token = "[SEP]"
    input_sentences = df.apply(lambda x: ''.join([x['nucleus'], separation_token, x['satellite']]), axis=1)

    # merge input sentence and labels onto one list (to form dataset object later)
    data = []
    for text in input_sentences:
        datapoint = {'text': text}
        data.append(datapoint)
    data = np.array(data)
    dataset = datasets.Dataset.from_list(list(data))

    return dataset

def get_model(model_path):
    if 'tokenizer' not in locals(): # prevent accidental re-run of cell
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if 'model' not in locals(): # prevent accidental re-run of cell
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda')
    model.eval()

    return model, tokenizer

def tokenize_function(dataset):
    return tokenizer(dataset["text"], padding=True, truncation=True, return_tensors='pt')

def label_relations(df, model, tokenizer, label_encoder):
    # get data
    batch_size = 32
    dataset = prep_data(df)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pred_labels = []
    with torch.no_grad():
        for batch in dataloader:
            tokens = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt').to('cuda')
            output = model(**tokens)
            logits = torch.Tensor.cpu(output.logits)
            pred_labels.extend(np.argmax(logits, axis=-1).tolist())

    preds = label_encoder.inverse_transform(pred_labels)
    df['new_relation'] = preds

    # calculate porportion of changed labels
    diff = df.apply(lambda x: x['original_relation'] != x['new_relation'], axis=1)
    print(diff.sum(), "changed relations out of", df.shape[0], '(' + str(round(float(diff.sum()/df.shape[0]), 2)) + ')')

    return df

def main():
    # start processing text
    # context_path = 'wikipedia_articles/economic_depression.txt'
    sample_path = '../data/sample/'
    article_name = 'article'
    context_path = sample_path + article_name

    with open(context_path, 'r') as f:
        raw_original_text = f.read()
        print("Read input file successfully!")

    # remove erronous characters
    raw_original_text = raw_original_text.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\u2013", "-").replace(u"\u2014", "-").replace(u"\u201C", "-").replace(u"\u201D", "-")
    # text_path = "../data/sample/article"
    # write_to_text_file(text_path, raw_original_text)

    # process new data, adjust paths if necessary
    pickle_path = sample_path + article_name + ".pickle"
    edu_path = sample_path + article_name + ".edus"

    tree_dict = get_tree_dict(pickle_path)
    root_id = get_root_id(tree_dict)
    edus = get_edus_from_file(edu_path)
    rst_tree = visualize_rst_tree(tree_dict, root_id, edus, get_edu_text=True)

    # print(rst_tree.show(stdout=False, sorting=False)) # uncomment to visualize RST tree
    # print(edus)

    segments = extract_segments(edus, tree_dict, root_id)
    df = pd.DataFrame(segments)
    df.shape

    # assembly the whole piece of text from EDUs, to ensure allignment
    original_text = ""
    for edu in edus:
        original_text += edu.strip() + ' '

    # map "Comparison" to "Join" since GUM does not contain "Comparison"
    for row in df[df['original_relation'] == 'Comparison'].iterrows():
        df.at[row[0], 'original_relation'] = "Joint"

    # get model
    model_path = "../models/output_dataset_correct_order_fold_1/checkpoint-23000"
    model, tokenizer = get_model(model_path)

    # get labels
    label_text = ['Attribution', 'Background', 'Cause', 'Condition', 'Contrast',
        'Elaboration', 'Enablement', 'Evaluation', 'Explanation', 'Joint',
        'Manner-Means', 'Same-Unit', 'Summary', 'Temporal',
        'Textual-Organization', 'Topic-Change', 'Topic-Comment']

    label_shorthand = ['Attr', 'Bckg', 'Cause', 'Cond', 'Contst',
        'Elab', 'Enab', 'Eval', 'Expl', 'Joint',
        'Man-Mean', 'Same-Un', 'Sum', 'Temp',
        'Text-Org', 'Top-Chang', 'Top-Com']

    le = LabelEncoder()
    le.fit(label_text)
    labels = le.transform(df.original_relation)
    
    # get relations
    df = label_relations(df, model, tokenizer, le)

    # fix old tree with new relations
    new_tree_dict = add_new_relations_to_tree_dict(tree_dict, df)
    new_rst_tree = visualize_rst_tree(new_tree_dict, get_root_id(new_tree_dict), edus, new_relation=True, get_edu_text=True)

    discourse_file_path = sample_path + article_name + "_discourse.csv" 
    df.to_csv(discourse_file_path, index=False)
    print(f"New relations saved successfully to '{discourse_file_path}'!")
    # print(new_rst_tree.show(stdout=False, sorting=False)) # for visualizing rst_tree
        
    # label_cnt = pd.Series(pred_labels).value_counts()
    # indices = pd.Series(label_shorthand).reindex(label_cnt.index, fill_value=0)

    # plt.figure(figsize=(12, 5))
    # plt.yticks(np.arange(1, max(label_cnt) + 1))
    # plt.bar(indices, label_cnt)

if __name__ == '__main__':
    main()