import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, T5ForConditionalGeneration
import re

MAX_DIS_SOURCE_LENGTH = 600
MAX_DIS_TARGET_LENGTH = 256
PREFIX = "make 3 distractors:"

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

def get_subtext(text, key, length):
    """Get subtext of text with length specified surrounding key.\
          0.4 of length will be before the key, 0.6 will be after. 

    Args:
        text (str): 
        length (str): 
    Returns:
        str: the subtext
    """
    bef_len = int(0.4 * length)
    af_len = length - bef_len

    text_words = text.split()
    key_words = key.split()
    
    bef_words = text[:text.find(key)].split()
    pos_bef = len(bef_words)
    pos_af = pos_bef + len(key_words)

    start_ind = 0
    end_ind = len(text_words)

    remainder_start = 0
    remainder_end = 0

    # get start index to extract
    if pos_bef > bef_len:
        start_ind = pos_bef - bef_len
    else:
        start_ind = 0
        remainder_start = bef_len - pos_bef

    # get end index to extract
    if pos_af + af_len <= len(text_words):
        end_ind = pos_af + af_len
    else:
        end_ind = len(text_words)
        remainder_end = pos_af + af_len - len(text_words)

    # if both ends are enough or have remainders
    if (remainder_start != 0 and remainder_end != 0) or (remainder_start == 0 and remainder_end == 0):
        return ' '.join(text_words[start_ind:end_ind])

    # if only one of two ends have remainder
    if remainder_start == 0:
        start_ind = max(0, start_ind - remainder_end)
    if remainder_end == 0:
        end_ind = min(len(text_words), end_ind + remainder_start)

    return ' '.join(text_words[start_ind:end_ind])

def postprocess_distractor(dis):
    """Post process generated distractors.
    Pipeline: Remove model's tags, remove redundant spaces.

    Args:
        dis (str): generated distractor

    Returns:
        str: cleaned distractor
    """

    new_dis = dis
    special_tags = ['</s>', '<unk>', '<pad>']
    for tag in special_tags:
        new_dis = new_dis.replace(tag, '')

    new_dis = re.sub(r'\s+', ' ', new_dis)

    new_dis_c = list(new_dis)
    for i in range(len(new_dis_c)):
        if len(new_dis_c[i].strip()) == 0:
            continue
        new_dis_c[i] = new_dis_c[i].upper()
        new_dis_c = new_dis_c[i:]
        break
    return ''.join(new_dis_c)

def generate_3_distractors(model, tokenizer, context, question, answer):
    """Generate 3 distractors

    Args:
        context (str):
        question (str):
        answer (str):
    Return:
        Tuple(dis1, dis2, dis3)
    """

    inputs = tokenizer(text=f"{PREFIX} question: {question}, answer: {answer}, context: {context}",
                        max_length=MAX_DIS_SOURCE_LENGTH,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt').to('cuda')

    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=MAX_DIS_TARGET_LENGTH)[0]

    # find sep tokens, sep tokens separate among distractors
    output_sequences = [token for token in output_sequences if token not in tokenizer.convert_tokens_to_ids(['<pad>', '</s>'])]
    sep_ids = [i for i, v in enumerate(output_sequences) if v == tokenizer.convert_tokens_to_ids(['<sep>'])[0]]
    try:
        assert len(sep_ids) == 2
    except:
        print("Not enough seperation tokens were found!")
        return ("", "", "")

    # remove other special tokens
    dis1 = tokenizer.batch_decode([output_sequences[:sep_ids[0]]])[0]
    dis2 = tokenizer.batch_decode([output_sequences[sep_ids[0] + 1:sep_ids[1]]])[0]
    dis3 = tokenizer.batch_decode([output_sequences[sep_ids[1] + 1:]])[0]

    return (dis1, dis2, dis3)

def generate_distractors_for_questions(model, tokenizer, context, question_df):
    final_questions_and_distractors = []
    for r in question_df.iterrows():
        ques = r[1]['question']
        ans = r[1]['answer']
        nucleus = r[1]['nucleus']

        sub_context = get_subtext(context, nucleus, 500) # get 500 words
        dis1, dis2, dis3 = generate_3_distractors(model, tokenizer, sub_context, ques, ans)

        datapoint = r[1]
        datapoint['distractor1'] = dis1
        datapoint['distractor2'] = dis2
        datapoint['distractor3'] = dis3
        final_questions_and_distractors.append(datapoint)

    return pd.DataFrame(final_questions_and_distractors)    

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

    # get edus
    edu_path = sample_path + article_name + ".edus"
    edus = get_edus_from_file(edu_path)
    # assembly the whole piece of text from EDUs, to ensure allignment
    original_text = ""
    for edu in edus:
        original_text += edu.strip() + ' '

    # get questions
    questions_path = sample_path + article_name + "_questions.csv"
    questions_df = pd.read_csv(questions_path)

    # get distractor models
    model_path = "../models/t5_base_distractors_generation_with_synthesized_dataset_custom_loss_sep_token_max_len_600/checkpoint-24000"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to('cuda')

    questions_and_distractors = generate_distractors_for_questions(model, tokenizer, original_text, questions_df)

    # save
    save_path = sample_path + article_name +"_questions_and_distractors.csv"
    questions_and_distractors.to_csv(save_path, index=False)

    full_questions_file_path = sample_path + article_name +"_questions_and_distractors.txt"
    with open(full_questions_file_path, 'w') as f:
        f.write(f"Context: {original_text}\n")
        f.write("--------------------\n")

        for row in questions_and_distractors.iterrows():
            f.write("\nNucleus: " + row[1]['nucleus'] + '\n')
            f.write("Satellite: " + row[1]['satellite'] + '\n')

            f.write("\nRelation: " + row[1]['relation'] + '\n')
            f.write(f"\nQuestion: {row[1]['question']}")
            f.write(f"\nAnswer: {row[1]['answer']}")
            f.write(f"\nDistractor 1: {row[1]['distractor1']}")
            f.write(f"\nDistractor 2: {row[1]['distractor2']}")
            f.write(f"\nDistractor 3: {row[1]['distractor3']}\n")

    print(f"Distractors generated and saved to '{save_path}' and '{full_questions_file_path}'!")

if __name__ == '__main__':
    main()
