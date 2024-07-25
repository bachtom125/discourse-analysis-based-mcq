import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spacy
from spacy import displacy
import inflect
import math
import re
from fastcoref import FCoref, LingMessCoref
from transformers import T5Tokenizer, T5ForConditionalGeneration

RELATIVE_PRONOUNS = ['who', 'that', 'whose', 'which']
AUXILIARY_SHORTHAND = {
    "'s": "is",
    "'re": "are",
    "'ve": "have",
    "'d": "had",
    "'ll": "will",
    "n't": "not"
}
DISCOURSE_MARKERS = [
  'accordingly', 'additionally', 'afterward', 'also',
  'although', 'as a final point', 'as a result', 'assuming that',
  'besides', 'but also', 'compared to', 'consequently', 'conversely', 'despite',
  'even though', 'finally', 'first', 'firstly', 'for example', 'for instance',
  'for the purpose of', 'furthermore', 'hence', 'however', 'if', 'importantly',
  'in addition', 'in case', 'in conclusion', 'in contrast', 'by contrast', 'in fact',
  'in order to', 'in other words', 'in the event that', 'in the same way',
  'indeed', 'just as', 'lastly', 'likewise', 'moreover', 'namely',
  'nevertheless', 'next', 'nonetheless', 'not only', 'of course', 'on condition that',
  'on the contrary', 'on the one hand', 'on the other hand', 'otherwise', 'plus', 'previously',
  'provided that', 'second', 'secondly', 'similarly', 'similarly to', 'since',
  'so', 'so long as', 'as long as', 'provided that', 'provided', 'so that', 'specifically', 'subsequently',\
  'such as', 'that is to say', 'that is'
  'then', 'therefore', 'third', 'thirdly', 'thus', 'to conclude', 'to illustrate',
  'to put it differently', 'to sum up', 'ultimately', 'undoubtedly', 'unless',
  'while', 'with the aim of', 'yet', 'then', 'and then', 'and'
  'as a consequence', 'as a result', 'in which', "at which", "where", "followed by", "following"
]

p = inflect.engine()
nlp = spacy.load('en_core_web_trf')
coref_resolver = LingMessCoref(device='cuda:0')

def get_subj(clause, accept_pron=True, accept_expl=False):
    """Return subject of clause, None of none found

    Args:
        clause (str):
        accept_expl (bool, optional): If take expletive as subject. Defaults to False.
        accept_expl (bool, optional): If take pronoun as subject. Defaults to True.

    Returns:
        str: The subject
    """
    doc = nlp(clause)

    for token in doc:
        if 'nsubj' in token.dep_:
            if token.pos_ == "PRON" and not accept_pron:
                continue
            for chunk in doc.noun_chunks:
                if chunk.start <= token.i and token.i < chunk.end:
                    return chunk.text
        if accept_expl:
            if 'expl' in token.dep_:
                return token.text
    return None

def get_main_verb(clause):
    """Get main verb of a clause, return None if none found.

    Args:
        clause (str):
    Returns:
        str:
    """
    doc = nlp(clause)
    root_tok = list(doc.sents)[-1].root

    return root_tok.text

def get_last_noun_chunk(clause):
    """Get last noun chunk of the clause, return None if none found.

    Args:
        clause (str):
    Returns:
        str:
    """
    doc = nlp(clause)
    if len(list(doc.noun_chunks)) != 0:
        last_noun_chunk = list(doc.noun_chunks)[-1]

        return last_noun_chunk.text
    return None

def check_relative_clause(text):
    """Return if the text (has to contain only one clause) is a relative clause,
    relative clauses can start with "which", "Ving, "Ved" (not including adverbial clause)

    Args:
        text
    Return
        boolean
    """
    doc = nlp(text)

    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if ("nsubj" in token.dep_):
                if (token.text.lower() in RELATIVE_PRONOUNS):
                    return 1 # relative clause starting with relative pronouns
                break

    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if (token.pos_ == "VERB") and (token.tag_ in ['VBG', 'VBN']):
                if len(token.morph.get('Tense')) == 0:
                    return 0 # not relative clause
                if token.dep_ == "ROOT" and ((token.text.strip().endswith('ing')) or (token.tag_ == 'VBN')):
                    return 2 # shortened relative clause (starting with Ving or Ved)
            else:
                return 0

def find_boundary(sent_doc, text_doc):
    """Find boundary indices of text_doc in sent_doc (given that text_doc is in sent_doc)

    Args:
        sent_doc (nlp Doc):
        text_doc (nlp Doc):
    """
    start_ind = 0
    while start_ind < len(sent_doc):
        if text_doc.text not in sent_doc[start_ind:].text:
            break
        start_ind += 1
    start_ind -= 1

    end_ind = len(sent_doc)
    while end_ind > 0:
        if (text_doc.text not in sent_doc[:end_ind].text) or (end_ind <= start_ind):
            break
        end_ind -= 1
    end_ind += 1

    return start_ind, end_ind

def is_dependent_clause(src_sent, text):
    sent_doc = nlp(src_sent)
    text_doc = nlp(text)

    start_ind, end_ind = find_boundary(sent_doc, text_doc)
    if start_ind < 0 or end_ind > len(sent_doc):
        print("\nText not found in source sentence!\n")
        return None

    for token in sent_doc[start_ind:end_ind]:
        if token.dep_ in ['acl', 'advcl', 'xcomp']: # omitted "ccomp", put back again if needed
            if token.head.i in range(start_ind, end_ind):
                return False
            return True
    return False

def unshorten_relative_clause(original_sent, clause):
    """Unshorten relative clause (make sure it's a relative clause before calling this method) ending with Ving or Ved, convert them to which/who + V

    Args:
        clause (str): relative clause containing Ving or Ved
        original_sent (str): original sentence containing that clause
    Return:
        tuple(str, str): modified clause and source sentence
    """

    clause_doc = nlp(clause)
    text_doc = nlp(original_sent)
    vb = ""

    for token in clause_doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if (token.pos_ == "VERB") and (token.tag_ in ['VBG', 'VBN']): # finds Ving or Ved
                vb = token

    for v_token in text_doc:
        # find verb in original sentence
        if v_token.text == vb.text.strip():
            c_i = vb.i
            t_i = v_token.i
            is_verb = True
            while c_i < len(clause_doc) and t_i < len(text_doc):
                if clause_doc[c_i].text.strip() != text_doc[t_i].text.strip():
                    is_verb = False
                    break
                c_i += 1
                t_i += 1
            if is_verb:
                break

    pointed_noun = v_token.head
    pointed_noun_chunk = None
    for chunk in text_doc.noun_chunks:
        if chunk.start <= pointed_noun.i and pointed_noun.i < chunk.end:
            pointed_noun_chunk = chunk

    if pointed_noun_chunk:
        pointed_root_noun = pointed_noun_chunk.root
    else:
        pointed_root_noun = pointed_noun

    rel_pro = "which"
    # not a fool-proof way to determine if noun is person
    if pointed_root_noun.ent_type_:
        if pointed_root_noun.ent_type_ == "PERSON":
            rel_pro = "who"

    # check plurality of noun/pronoun and conjugate accordingly
    # only applicable for Present Tense, not for past or others
    if pointed_root_noun.pos_.startswith("NOUN"): # noun
        plurality = pointed_root_noun.tag_ == "NNS"
    elif "PRON" in pointed_root_noun.pos_:  # pronoun
        plurality = (pointed_root_noun.lemma_ == "we") or (pointed_root_noun.lemma_ == "you") or (pointed_root_noun.lemma_ == "they") or ((pointed_root_noun.lemma_ == "I"))
    else: 
        plurality = False # temporary solution

    if not plurality:
        if vb.tag_ == 'VBG':
            conj_vb = p.plural_noun(vb.lemma_) # get singular conjugation (plural_noun() method works with verbs too)
        else:
            if pointed_root_noun.text.strip() == "I":
                aux = 'am'
            else:
                aux = 'is'
            conj_vb = aux + ' ' + vb.text
    else:
        if vb.tag_ == 'VGB':
            conj_vb = p.plural_verb(vb.lemma_) # get plural conjugation
        else:
            if pointed_root_noun.text.strip() == "I":
                aux = 'am'
            else:
                aux = 'are'
            conj_vb = aux + ' ' + vb.text

    fixed_clause = clause.replace(vb.text, rel_pro + ' ' +  conj_vb, 1)# replace only the first occurence of the verb
    fixed_sent = original_sent.replace(clause.strip(), fixed_clause.strip(), 1)
    return  (fixed_sent, fixed_clause)

def is_one_clause(text, count_relative_clause=True):
    """Check if input text is one clause or multiple.
        NOTE: If not multiple clause, the method returns true, so does not account for the case of not a full clause, just check whether multiple clauses or not, cause a EDU is usually at least a clause semantically.

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    doc = nlp(text)

    # check how many subjects
    has_subj = False
    for token in doc:
        if "nsubj" in token.dep_:
            if not count_relative_clause:
                if token.text.lower() in ['who', 'whom', 'whose', 'which', 'that']:
                    continue
                if token.head.dep_ in ['relcl', 'acl', 'ccomp']:
                    continue
            if has_subj:
                return False
            else:
                has_subj = True
    return True # not return has_subj, so that even no subject will be 1 clause

def is_single_sentence(text):
    """Determine if input text is one single sentence (one main subject and one main verb)

    Args:
        text (str):

    Returns:
        Boolean:
    """
    # Parse the input text
    doc = nlp(text)

    # Initialize flags for subject and main verb
    subject_count = 0
    verb_count = 0

    # Iterate through the tokens in the parsed document
    for token in doc:
        # Check for a subject
        if "nsubj" in token.dep_:
            subject_count += 1
        # Check for the main verb
        if token.pos_ == "VERB" and token.dep_ not in ["pcomp", "relcl", "acl", "ccomp"]:
            verb_count += 1

    # Determine if the text is a single clause
    return verb_count == 1 and subject_count == 1

def has_aux(sentence):
    """Check if sentence has auxiliary verb.
    Input one sentence only.
    """
    doc = nlp(sentence)
    for token in doc:
        if "AUX" in token.pos_:
            return True
    return False

def move_aux_to_beginning(sentence):
    """Move auxiliary verb to the beginning of sentence (to form question).
    Input one sentence only. Make sure it has aux verb. Make sure sentence starts with subject.
    """
    doc = nlp(sentence)
    aux = ""
    for token in doc:
        if "AUX" in token.pos_:
            aux = token.text
            break
    assert len(aux)

    if aux.strip() in AUXILIARY_SHORTHAND:
        new_aux = AUXILIARY_SHORTHAND[aux]
        new_sent = new_aux + ' ' + sentence.strip().replace(aux, '', 1).replace(sentence[0], sentence[0].lower(), 1)
    else:
        new_sent = aux + ' ' + sentence.strip().replace(aux, '', 1).replace(sentence[0], sentence[0].lower(), 1)

    return new_sent

# move aux test run
def has_verb(sentence):
    """Check if sentence has normal verb.
    Input one sentence only.
    """
    doc = nlp(sentence)
    for token in doc:
        if "VERB" in token.pos_:
            return True
    return False

def choose_and_replace_aux_for_verb(sentence):
    """Put appropriate aux at beginnging of clause
    """
    doc = nlp(sentence)
    main_verb = None

    for token in doc:
        if "VERB" in token.pos_:
            main_verb = token
            break

    if not main_verb:
        print("\nCan't find main verb in", sentence, '!\n')
        return ""

    # check plurality
    pointed_noun = main_verb.head
    if pointed_noun.pos_.startswith("NOUN"): # noun
        plurality = pointed_noun.tag_ == "NNS"
    elif pointed_noun.pos_.startswith("PRP"):  # pronoun
        plurality = (pointed_noun.lemma_ == "we") or (pointed_noun.lemma_ == "you") or (pointed_noun.lemma_ == "they")
    else: 
        plurality = True # temporary solution

    # check tense
    tense = "present" if main_verb.tag_ in ['VBZ', 'VBP'] else 'past'
    # get aux
    aux = {
      "present": "do" if plurality else "does",
      "past": "did",
    }.get(tense)

    # replace appropriate auxilary
    new_sent = sentence.replace(main_verb.text, main_verb.lemma_, 1)
    new_sent = aux + ' ' + new_sent

    return new_sent

def remove_ending_special_chars(sentence):
    """Remove ending non-word characters of a sentence

    Args:
        sentence (str): sentence to be stripped

    Returns:
        str: stripped sentence
    """
    sen_len = len(sentence)
    for i in range(sen_len - 1, -1, -1):
        char = sentence[i]

        # check if the character is a punctuation mark
        if char.isalnum():
            return sentence
        else:
            sentence = sentence[:i]
    return sentence.strip()

def remove_leading_special_chars(sentence):
    """Remove leading non-word characters of a sentence

    Args:
        sentence (str): sentence to be stripped

    Returns:
        str: stripped sentence
    """

    start_ind = 0
    for i in range(0, len(sentence)):
        char = sentence[i]
        # check if the character is a punctuation mark
        if char.isalnum():
            break
        else:
            start_ind = i + 1

    return sentence[start_ind: ].strip()

def split_into_sentences(text):
    """Split text into sentence

    Args:
        text (str): text to be splited
    Return:
        list[str]: split text
    """
    sents = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)", text)
    sents = [sent for sent in sents if len(sent.strip())]
    return sents

def split_into_paras(text, deli):
    paras = text.split(deli)
    paras = [para.strip() for para in paras if len(para.strip()) != 0]
    return paras

def get_para_id(paras, text):
    """Find paragraph index of text

    Args:
        paras (list[str]): list of paragraphs
        text (str): text to find
    Return:
        int: para id, -1 if not found
    """

    for p_i in range(len(paras)):
        if paras[p_i].find(text.strip()) != -1:
            return p_i
    return -1

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

def find_source_sents(sents, text, span=2):
    """Find the sentences of which the text is a part.

    Args:
        sents (str):
        text (str):
        span (int): span of sentences to the left to return
    """
    start_ind = 0
    while start_ind < len(sents):
        if text not in ''.join(sents[start_ind:]):
            break
        start_ind += 1
    start_ind -= 1

    end_ind = len(sents) # exclusive
    while end_ind > 0:
        if (text not in ''.join(sents[:end_ind])) or (end_ind <= start_ind):
            break
        end_ind -= 1
    end_ind += 1

    if (start_ind < 0) or (end_ind > len(sents)):
        print(f"\nCan't find source sentences of {text}. Returning the text\n")
        return text

    src_sents = ''.join(sents[max(start_ind - span, 0):end_ind])
    return src_sents

def add_subject_to_relative_clause(original_sent, clause):
    """Find and Prepend (with modifications) the subject of relative clause that does not contain one

    Args:
        original_sent (str): sentence from which the clause is extracted
        clause (str): clause for which to find subject
    Return:
        str: subject
    """
    doc = nlp(original_sent)
    subj = ""
    clause_start_ind = original_sent.find(clause)
    for token in doc:
        if (len(subj)):
            break
        if (token.dep_ in ['relcl', 'acl']) and (token.idx >= clause_start_ind) and (token.idx < (clause_start_ind + len(clause))): # relative clause is noun modifier
            for chunk in doc.noun_chunks:
                if token.head.i >= chunk.start and token.head.i < chunk.end:
                    subj = chunk.text
                    break

        # 'cause adverbial clauses are often in relations that do not require unshortening of clause
        if (token.dep_ in ['advcl', 'ccomp']) and (token.idx >= clause_start_ind) and (token.idx < (clause_start_ind + len(clause))):
            return None
            # for chunk in doc.noun_chunks:
            #     if token.head.i >= chunk.start and token.head.i < chunk.end:
            #         subj = chunk.text
            #         break

    if not len(subj): # not found subject
        print("\nCan't find subject of", original_sent, "!\n")
        return None

    for token in doc:
        if ("nsubj" in token.dep_) and (token.text.lower() in RELATIVE_PRONOUNS):
            new_clause = clause.replace(token.text, subj) # contains more nuances (where -> in + N, which -> N, who -> N)
            new_sent = original_sent.replace(clause.strip(), new_clause.strip(), 1)
            return new_sent, new_clause

def add_subject_to_fragmented_clause(original_sent, clause):
    """Find and Prepend (with modifications) the subject of a clause that does not contain one, the clause has to contain a verb.

    Args:
        original_sent (str): sentence from which the clause is extracted
        clause (str): clause for which to find subject
    Return:
        str: subject
    """
    doc = nlp(original_sent)
    clause_doc = nlp(clause)

    # get main verb of clause
    root_token = None
    for token in clause_doc:
        if "nsubj" in token.dep_:
            return original_sent, clause

        if token.dep_ == "ROOT" and "VERB" in token.pos_:
            root_token = token
            break
    if root_token is None:
        return original_sent, clause

    subj = None
    main_verb = None
    clause_start_ind = original_sent.find(clause)
    for token in doc:
        if (token.text == root_token.text) and (token.idx >= clause_start_ind) and (token.idx < (clause_start_ind + len(clause))):
            main_verb = token
            # token found here is verb of clause, now get verb of the missing part of the clause
            other_verb = token.head
            for c in other_verb.children:
                if "nsubj" in c.dep_:
                    subj = c
                    break

    if subj is None or main_verb is None:
        print(f"\nCan't find subject for {clause}. Return the original clause.\n")
        return original_sent, clause

    for chunk in doc.noun_chunks:
        if subj.i >= chunk.start and subj.i < chunk.end:
            subj = chunk
            break

    subj = subj.text
    subj_char = list(subj)
    subj_char[0] = subj_char[0].lower()
    subj = ''.join(subj_char)

    # get starting point to replace, have main_verb, could be aux before main verb
    for child in main_verb.children:
        if 'aux' in child.dep_:
            main_verb = child
            break

    new_clause = clause.replace(main_verb.text, subj + ' ' + main_verb.text)
    new_sent = original_sent.replace(clause.strip(), new_clause.strip(), 1)

    return new_sent, new_clause

def remove_leading_conjunction(original_sent, text):
    """Remove leading conjunctions from text, return intact if cannot find any

    Args:
        original_sent (_type_): _description_
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    doc = nlp(text)

    conj = ""
    for token in doc:
        if token.pos_ in ['PUNCT', 'SYM']:
            continue
        if "CONJ" not in token.pos_:
            return (original_sent, text)
        conj = token.text
        break

    new_text = text.replace(conj.strip(), '', 1)
    new_sent = original_sent.replace(text, new_text)

    return (new_sent, new_text)

def remove_leading_adverb(original_sent, text):
    """Remove leading adverb from text, return intact if cannot find any

    Args:
        original_sent (_type_): _description_
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    doc = nlp(text)
    if text is None:
        print("This is fucking it: ", original_sent)

    for token in doc:
        if token.pos_ in ['PUNCT', 'SYM']:
            continue
        if token.pos_ != 'ADV':
            return (original_sent, text)
        adv = token.text
        break

    new_text = text.replace(adv.strip(), '', 1)
    new_sent = original_sent.replace(text, new_text)

    return (new_sent, new_text)

def remove_leading_discourse_marker(original_sent, text):
    """Remove leading discourse markers from text, return intact if cannot find any

    Args:
        original_sent (_type_): _description_
        text (_type_): _description_

    Returns:
        _type_: _description_
    """

    discourse_marker = None
    min_pos = len(text)
    for dm in DISCOURSE_MARKERS:
        find_result = text.lower().find(dm)
        if find_result > -1:
            if find_result < min_pos:
                min_pos = find_result
                discourse_marker = dm
            if find_result == min_pos and len(dm) > len(discourse_marker):
                discourse_marker = dm

    # check if found marker stand at the beginning of text

    if discourse_marker is not None:
        for i in range(0, min_pos):
            if text[i].isalnum() or text[min_pos + len(discourse_marker)].isalnum():
                discourse_marker = None
                break

    if not discourse_marker:
        return (original_sent, text)

    pattern = re.compile(discourse_marker, re.IGNORECASE)

    new_text = pattern.sub("", text, 1)
    new_sent = original_sent.replace(text, new_text)

    return (new_sent, new_text)

# to be integrated

def replace_substr(text, substring, start_ind, end_ind):
  """Replace part of text with specified index [start_ind, end_ind) with substring
  """
  try:
    assert(len(substring)  == end_ind - start_ind)
  except:
    print('Text:', text, '--', sep='')
    print('Substring:', substring, '--', sep='')

  text_l = list(text)
  text_l[start_ind:end_ind] = list(substring)

  return ''.join(text_l)

def is_in_ref(index, clusters):
    """Check if current index is in one of the references

    Args:
        index (int): index to check
        clusters (list(list(tuple))): list of cluster, each cluster containing a list of tuple correponding to indices of the references
    Return:
        tuple (verdict, (start, end), (ref_token_start, ref_token_end)): -1 both index if not found, ref_token is token to relace
    """
    for cluster in clusters:
        for token in cluster:
            if cluster.index(token) == 0:
                continue
            if index >= token[0] and index < token[1]:
                ref_token = (cluster[0][0], cluster[0][1])
                return True, token, ref_token

    return False, (-1, -1), (-1, -1)

def check_relative_clause_type(clause):
    """Check which type of relative clause "clause" is:
    - which + V + clause (present subject is sufficient for being clause)
    - which + V + (not clause)

    Args:
        sent (str):
        clause (str):
    Return: 0 or 1, -1 if not relative clause expected (not start with which + V), then it could be an adverbial clause
    """

    doc = nlp(clause.strip())

    i = 0
    while i < len(doc):
        token = doc[i]
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if "PRON" not in token.pos_:
                return -1
            else:
                break
        i += 1

    i += 1
    while i < len(doc):
        token = doc[i]
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if "VERB" not in token.pos_ and "AUX" not in token.pos_:
                return -1
            else:
                break
        i += 1

    # if reach here, is expected relative clause type (which + V)
    i += 1
    t_i = i # use to this to check subject and not modify i
    while t_i < len(doc):
        token = doc[t_i]
        if 'nsubj' in token.dep_:
            return 0
        t_i += 1

    return 1

def resolve_coreference(original_sent, text):
    """Perfrom coreference resolution and replace corresponding text.

    Args:
        original_sent (str): Sentence the text was derived froms
        text (str): Target text
    """
    
    text_ind = original_sent.find(text) # starting index of text in original text
    coref_preds = coref_resolver.predict(texts=[original_sent])
    coref_clusters = coref_preds[0].get_clusters(as_strings=False)
    new_sent = []
    new_text = []

    # interate string left to right while appending current char to a new list
    # if current index in one of the token in one of the clusters, add the replacement to the list, keep the text intact, to know what index are at
    i = 0

    # if referred word is a verb, use have to notice and discard the sentence.
    while i < len(original_sent):
        find_result = is_in_ref(i, coref_clusters)
        if find_result[0]:
            token = find_result[1]
            token_ref = find_result[2]
            new_sent.append(original_sent[token_ref[0]:token_ref[1]])
            if (i >= text_ind and i < text_ind + len(text)):
                if (text_ind <= token[0] and token[1] <= text_ind + len(text)):
                    new_text.append(original_sent[token_ref[0]:token_ref[1]])
                else:
                    new_text.append(original_sent[i:text_ind + len(text)])
            i = token[1]
        else:
            new_sent.append(original_sent[i])
            if i >= text_ind and i < text_ind + len(text):
                new_text.append(original_sent[i])
            i += 1

    return ''.join(new_sent), ''.join(new_text)

def remove_relative_pronoun(original_sent, text):
    """Remove relative pronouns from text, make sure it's a relative clause first

    Args:
        original_sent (str):
        text (str):
    """
    doc = nlp(text)
    rel_pro = ""
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if ("nsubj" in token.dep_):
                if (token.text.lower() in RELATIVE_PRONOUNS):
                    rel_pro = token.text

    if len(rel_pro) == 0:
        return original_sent, text

    new_text = text.replace(rel_pro.strip(), '', 1).strip()
    new_sent = original_sent.replace(text, new_text).strip()

    return new_sent, new_text

def preprocessing_pipeline(original_text, text, retain_dm=False, retain_conj=False, retain_adv=False, add_subject_to_rel_clause=False, resolve_coref=False, unshorten_rel=False, remove_rel_pron=True, get_full_sentence=True):
    """Perform necessary transformation steps.

    Args:
        text (str): raw text
    Return:
        (str): full clause from text
    """
    sents = split_into_sentences(original_text)
    text = text.strip()

    # coreferen ce resolution
    src_3_sents = find_source_sents(sents, text, 3)
    if resolve_coref:
        src_3_sents, text = resolve_coreference(src_3_sents, text)
    src_sent = find_source_sents(split_into_sentences(src_3_sents), text, 0)

    # if no source sentence found
    if src_sent is None:
        return "", text # system just use text, not original_sent anyway

    # removal of irrelavent components
    if not retain_dm:
        src_sent, text = remove_leading_discourse_marker(src_sent, text)
    if not retain_conj:
        src_sent, text = remove_leading_conjunction(src_sent, text)
    if not retain_adv:
        src_sent, text = remove_leading_adverb(src_sent, text)
    src_sent = remove_leading_special_chars(src_sent)
    text = remove_leading_special_chars(text)
    # src_3_sents = remove_ending_special_chars(src_3_sents)
    # text = remove_ending_special_chars(text)

    # handle single relative clause
    if is_one_clause(text, count_relative_clause=False):
        cl_type = check_relative_clause(text)
        if cl_type != 0: # is relative clause
            if cl_type == 2:
                if unshorten_rel:
                    src_sent, text = unshorten_relative_clause(src_sent, text)
            if add_subject_to_rel_clause:
                if add_subject_to_relative_clause(src_sent, text):
                    src_sent, text = add_subject_to_relative_clause(src_sent, text)
            if remove_rel_pron:
                src_sent, text = remove_relative_pronoun(src_sent, text)
        # is not relative clause:
        elif not is_single_sentence(text):
            if get_full_sentence:
                src_sent, text = add_subject_to_fragmented_clause(src_sent, text)

    return src_sent, text

# preprocessing_pipeline(original_text, "and thus cannot be within the thinking thing.")
def postprocess_question(ques):
    """Clean up the question after being generated.
    Pipeline: Clean double spaces, clean extra punctuations

    Args:
        ques (str): generated question
    Returns:
        str: new question
    """

    puncts_to_remove = ['.', ',', '!']

    ques_c = list(ques)
    for i in range(len(ques_c) - 1, 0, -1):
        if ques_c[i].isalnum():
            break

        if ques_c[i] in puncts_to_remove:
            ques_c.pop(i)

    # not adding extra space before question mark, used to have to for the complete question model to run well, but now use only keyword, not incomplete questions anymore
    # for i in range(len(ques_c) - 1, 0, -1):
    #     if ques_c[i] == '?':
    #         ques_c.insert(i, ' ')
    #         break

    new_ques = ''.join(ques_c)
    new_ques = re.sub(r'\s+', ' ', new_ques)
    return new_ques

def postprocess_answer(ans):
    """Clean up the answer after being generated.
    Pipeline: Clean double spaces, clean extra punctuations, capitalize first word.

    Args:
        ans (str): generated answer
    Returns:
        str: new answer
    """

    ending_puncts = ['.', '!']
    ans = ans.strip()

    ans_c = list(ans)
    has_punct = False
    for i in range(len(ans_c) - 1, 0, -1):
        if ans_c[i].isalnum():
            break

        if ans_c[i] in ending_puncts:
            if not has_punct:
                has_punct = True
                continue
        ans_c.pop(i)

    if not has_punct:
        ans_c.append('.')

    for i in range(len(ans_c)):
        if len(ans_c[i].strip()) == 0:
            continue
        ans_c[i] = ans_c[i].upper()
        ans_c = ans_c[i:]
        break

    new_ans = ''.join(ans_c)
    new_ans = re.sub(r'\s+', ' ', new_ans)
    return new_ans

"""Question Templates"""
def cause_question_type_0(nucleus, satellite): 
    """Make question based on CAUSE relationship
    Type 0: satellite (result) is: relative clause: which + verb + clause (e.g. which made him happy.)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """
    # experimenting:exactly the same as type_1
    doc = nlp(satellite)
    rel_pro = ""
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if "PRON" in token.pos_:
                rel_pro = token.text
                break

    assert len(rel_pro) > 0
    question = satellite.replace(rel_pro.strip(), "What", 1) + '?'
    answer = nucleus.strip() + '.'

    return (question, answer)


def cause_question_type_1(nucleus, satellite):
    """Make question based on CAUSE relationship
    Type 1: satellite (result) is: relative clause: which + verb + (not clause) (e.g. which caused the noise.)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """
    doc = nlp(satellite)
    rel_pro = ""
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if "PRON" in token.pos_:
                rel_pro = token.text
                break

    assert len(rel_pro) > 0
    question = satellite.replace(rel_pro.strip(), "What", 1) + '?'
    answer = nucleus.strip() + '.'

    return (question, answer)

def cause_question_type_2(nucleus, satellite):
    """Make question based on CAUSE relationship
    Type 2: satellite (result) is: full clause (not relative clause)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """

    if has_aux(satellite):
        new_sate = move_aux_to_beginning(satellite)
    else:
        new_sate = choose_and_replace_aux_for_verb(satellite)
    question = "Why " + new_sate.strip() + '?'
    answer = nucleus.strip() + '.'

    return (question, answer)

MAX_INC_SOURCE_LENGTH = 600
MAX_INC_TARGET_LENGTH = 128
INC_PREFIX = "make question:"

def complete_question(context, key, answer):
    """Complete question if needed, needs globally available model and tokenizer ("model" and "tokenizer")

    Args:
        context (str): 
        question (str): 
        answer (str): 
    Return:
        str: new question
    """
    model_path = "../models/t5_base_incomplete_questions_wkdiswfla_max_len_600/not_filter_long_contexts/checkpoint-66000"

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to('cuda')
    model.eval()

    inputs = tokenizer(text=f"{INC_PREFIX} answer: {answer}, key: {key}, context: {context}",
                            max_length=MAX_INC_SOURCE_LENGTH,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt').to('cuda')
            
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=MAX_INC_TARGET_LENGTH
    )
    
    output = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[-1]

    return output

def generate_cause_question(context, nucleus, satellite, use_template=False):
    """Generate question based on Cause relation. \
        Nucleus being the Cause, which is the answer.

    Args:
        context (str):
        nucleus (str):
        satellite (str):

    Returns:
        Tuple(Boolean, Tuple(str, str)): Boolean dictates if questions are plausible. \
            The second tuple is the question-answer pair.
    """
    nucleus_pair = preprocessing_pipeline(context, nucleus,
                                          add_subject_to_rel_clause=False, resolve_coref=False)
    satellite_pair = preprocessing_pipeline(context, satellite, resolve_coref=False)
    # to feed to complete question model for full context
    # full_nucleus_pair = preprocessing_pipeline(context, nucleus, resolve_coref=True)
    full_satellite_pair = preprocessing_pipeline(context, satellite, resolve_coref=True)

    # if cannot process one of the two
    if nucleus_pair[1] is None or satellite_pair[1] is None:
        return (False, ("", ""))

    nucleus = nucleus_pair[1]
    satellite = satellite_pair[1]

    # if use template
    if use_template:
        if is_one_clause(satellite, count_relative_clause=False):
            cl_type = check_relative_clause(satellite)
            if cl_type != 0: # is relative clause
                rel_type = check_relative_clause_type(satellite)
                if rel_type == 0:
                    question, answer = cause_question_type_0(nucleus, satellite)
                    return (True, (question, answer))
                elif rel_type == 1:
                    question, answer = cause_question_type_1(nucleus, satellite)
                    return (True, (question, answer))
        else:
            question, answer =  ("", "")
            return (False, (question, answer))
        question, answer =  cause_question_type_2(nucleus, satellite)
        return (True, (question, answer))

    # not use template -> use neural question generator
    else:
        key = "Why"
        last_sate_noun_chunk = get_last_noun_chunk(full_satellite_pair[1])
        if last_sate_noun_chunk is not None:
            key = key + ' ' + last_sate_noun_chunk

        answer = nucleus
        question = complete_question(context, key, answer)

        question = postprocess_question(question)
        answer = postprocess_answer(answer)
        return (True, (question, answer))

# contrast relation: ask what's different between two subjects.
# retain discourse markers

def make_contrast_question_with_two_clauses(nucleus, satellite):
    n_subj = get_subj(nucleus)
    s_subj = get_subj(satellite)

    if (n_subj is None) or (s_subj is None):
        print("\nCan't find subject!\n")
        return ("", "")

    if n_subj.strip().lower() == s_subj.strip().lower():
        print("\nSame subjects for nucleus and satellite!\n")
        return ("", "")

    question = "What is the difference between " + n_subj + " and " + s_subj + '?'
    answer = nucleus.strip() + ' ' + satellite.strip()
    return (question, answer)

def generate_contrast_question(context, nucleus, satellite, use_template=False):
    """Generate question based on Contrast relation.

    Args:
        context (str):
        nucleus (str):
        satellite (str):

    Returns:
        Tuple(Boolean, Tuple(str, str)): Boolean dictates if questions are plausible. \
            The second tuple is the question-answer pair.
    """
    nucleus_pair = preprocessing_pipeline(context, nucleus, retain_dm=True,
                                          retain_conj=True, retain_adv=True,
                                          add_subject_to_rel_clause=True, resolve_coref=False)
    satellite_pair = preprocessing_pipeline(context, satellite, retain_dm=True,
                                          retain_conj=True, retain_adv=True,
                                          add_subject_to_rel_clause=False, resolve_coref=False)
    # to feed to complete question model for full context
    # full_nucleus_pair = preprocessing_pipeline(context, nucleus, retain_dm=True,
                                        #   retain_conj=True, retain_adv=True,
                                        #   add_subject_to_rel_clause=True, resolve_coref=True)
    # full_satellite_pair = preprocessing_pipeline(context, satellite, retain_dm=True,
                                        #   retain_conj=True, retain_adv=True,
                                        #   add_subject_to_rel_clause=True, resolve_coref=True)
    # if cannot process one of the two
    if nucleus_pair[1] is None or satellite_pair[1] is None:
        return (False, ("", ""))

    nucleus = nucleus_pair[1]
    satellite = satellite_pair[1]

    # if use template
    if use_template:
        if is_one_clause(nucleus, count_relative_clause=False) and is_one_clause(satellite, count_relative_clause=False):
            ques, ans = make_contrast_question_with_two_clauses(nucleus, satellite)
            return (True, (ques, ans))
        return (False, ("", "")) # more than 1 clause

    # not use template -> use neural question generator
    else:
        # check if subjects are different
        n_subj = get_subj(nucleus)
        s_subj = get_subj(satellite)

        if n_subj and s_subj:
            if (n_subj.strip().lower() == s_subj.strip().lower()):
                return (False, ("", ""))

        # currently take full_answer (to feed to complete question model) to be the same as answer, \
        # which is basically the same as original nucleus and satellite, with no context added, \
        # might change later if needed.

        # full_answer = full_nucleus_pair[1].strip() + ' ' + full_satellite_pair[1].strip()
        if context.find(nucleus) < context.find(satellite):
            full_answer = nucleus.strip() + ' ' + satellite.strip()
            answer = nucleus.strip() + ' ' + satellite.strip()
        else:
            full_answer = satellite.strip() + ' ' + nucleus.strip()
            answer = satellite.strip() + ' ' + nucleus.strip()
        question = complete_question(context, "How different", full_answer)

        question = postprocess_question(question)
        answer = postprocess_answer(answer)
        return (True, (question, answer))

# generate_contrast_question(original_text, nuc, sate, use_template=False)
# condition relation

def condition_question_type_0(nucleus, satellite):
    """Make question based on condition relationship
    Type 0: satellite (result) is: relative clause: which + verb + clause (e.g. which made him happy.)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """
    # experimenting:exactly the same as type_1
    doc = nlp(satellite)
    rel_pro = ""
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if "PRON" in token.pos_:
                rel_pro = token.text
                break

    assert len(rel_pro) > 0
    question = satellite.replace(rel_pro.strip(), "What condition", 1) + '?'
    answer = nucleus.strip() + '.'

    return (question, answer)

def condition_question_type_1(nucleus, satellite):
    """Make question based on condition relationship
    Type 1: satellite (result) is: relative clause: which + verb + (not clause) (e.g. which conditiond the noise.)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """
    doc = nlp(satellite)
    rel_pro = ""
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if "PRON" in token.pos_:
                rel_pro = token.text
                break

    assert len(rel_pro) > 0
    question = satellite.replace(rel_pro.strip(), "What condition", 1) + '?'
    answer = nucleus.strip() + '.'

    return (question, answer)

def condition_question_type_2(nucleus, satellite):
    """Make question based on condition relationship
    Type 2: satellite (result) is: full clause (not relative clause)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """

    if has_aux(satellite):
        new_sate = move_aux_to_beginning(satellite)
    else:
        new_sate = choose_and_replace_aux_for_verb(satellite)
    question = "In what condition " + new_sate.strip() + '?'
    answer = nucleus.strip() + '.'

    return (question, answer)

def generate_condition_question(context, nucleus, satellite, use_template=False):
    """Generate question based on Condition relation.\
    Nucleus is the answer.

    Args:
        context (str):
        nucleus (str):
        satellite (str):

    Returns:
        Tuple(Boolean, Tuple(str, str)): Boolean dictates if questions are plausible. \
            The second tuple is the question-answer pair.
    """
    nucleus_pair = preprocessing_pipeline(context, nucleus, retain_dm=True,
                                          retain_conj=True, retain_adv=True,
                                          add_subject_to_rel_clause=True, resolve_coref=False)
    satellite_pair = preprocessing_pipeline(context, satellite, resolve_coref=True)
    # to feed to complete question model for full context
    # full_nucleus_pair = preprocessing_pipeline(context, nucleus, retain_dm=True,
    #                                       retain_conj=True, retain_adv=True,
    #                                       add_subject_to_rel_clause=True, resolve_coref=True)
    # full_satellite_pair = preprocessing_pipeline(context, satellite, retain_dm=True,
    #                                       retain_conj=True, retain_adv=True,
    #                                       add_subject_to_rel_clause=True, resolve_coref=True)
    # if cannot process one of the two
    if nucleus_pair[1] is None or satellite_pair[1] is None:
        return (False, ("", ""))

    # extract nucleus and satellite
    nucleus = nucleus_pair[1]
    satellite = satellite_pair[1]

    # if use template
    if use_template:
        if is_one_clause(satellite, count_relative_clause=False):
            cl_type = check_relative_clause(satellite)
            if cl_type != 0: # is relative clause
                rel_type = check_relative_clause_type(satellite)
                if rel_type == 0:
                    ques, ans =  condition_question_type_0(nucleus, satellite)
                    return (True, (ques, ans))
                elif rel_type == 1:
                    ques, ans =  condition_question_type_1(nucleus, satellite)
                    return (True, (ques, ans))
            else: # 1 clause, not relative clause
                ques, ans =  condition_question_type_2(nucleus, satellite)
                return (True, (ques, ans))
        else:
            return (False, ("", ""))

    # not use template -> use neural question generator
    else:
        # currently take full_answer (to feed to complete question model) to be the same as answer, \
        # which is basically the same as original nucleus, with no context added, \
        # might change later if needed.

        # full_answer = full_satellite_pair[1].strip()
        full_answer = satellite.strip()

        question = complete_question(context, "Under what condition", full_answer)
        answer = satellite.strip()

        question = postprocess_question(question)
        answer = postprocess_answer(answer)
        return (True, (question, answer))

# enablement relation
# difference to manner-means: enablement encapsulate manner-means, as all means can "enable" the goal,
# but enablement also contains situational aid, an event lead (may not intentionally) to another event

# enablement relation
def enablement_question_type_0(nucleus, satellite):
    """Make question based on enablement relationship
    Type 0: satellite (result) is: relative clause: which + verb + clause (e.g. which made him happy.)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """
    # experimenting: exactly the same as type_1
    doc = nlp(satellite)
    rel_pro = ""
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if "PRON" in token.pos_:
                rel_pro = token.text
                break

    assert len(rel_pro) > 0
    question = satellite.replace(rel_pro.strip(), "What ", 1) + '?'
    answer = nucleus.strip() + '.'

    return (question, answer)

def enablement_question_type_1(nucleus, satellite):
    """Make question based on enablement relationship
    Type 1: satellite (result) is: relative clause: which + verb + (not clause) (e.g. which enablementd the noise.)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """
    doc = nlp(satellite)
    rel_pro = ""
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if "PRON" in token.pos_:
                rel_pro = token.text
                break

    assert len(rel_pro) > 0
    question = satellite.replace(rel_pro.strip(), "What ", 1) + '?'
    answer = nucleus.strip() + '.'

    return (question, answer)

def enablement_question_type_2(nucleus, satellite):
    """Make question based on enablement relationship
    Type 2: satellite (result) is: full clause (not relative clause)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """

    if has_aux(satellite):
        new_sate = move_aux_to_beginning(satellite)
    else:
        new_sate = choose_and_replace_aux_for_verb(satellite)
    question = "How " + new_sate.strip() + '?'
    answer = nucleus.strip() + '.'

    return (question, answer)

def enablement_question_type_3(nucleus, satellite):
    """Make question based on enablement relationship
    Type 2: satellite (result) is: relative clause but not start with relative pronoun (most probably adverbial clause)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """

    question = "What can be done " + satellite.strip() + '?'
    answer = nucleus.strip() + '.'

    return (question, answer)

def generate_enablement_question(context, nucleus, satellite, use_template=False):
    """Generate question based on Enablement relation.\
    Nucleus (the enabler) forms the question. f
    Satellite (the enablee) is the answer.

    Args:
        context (str):
        nucleus (str):
        satellite (str):

    Returns:
        Tuple(Boolean, Tuple(str, str)): Boolean dictates if questions are plausible. \
            The second tuple is the question-answer pair.
    """
    nucleus_pair = preprocessing_pipeline(context, nucleus, add_subject_to_rel_clause=True,
                                          resolve_coref=True)
    satellite_pair = preprocessing_pipeline(context, satellite, retain_dm=False,
                                          retain_conj=True, retain_adv=True,
                                          add_subject_to_rel_clause=True, resolve_coref=False, unshorten_rel=True, get_full_sentence=False)
    # to feed to complete question model for full context
    full_nucleus_pair = preprocessing_pipeline(context, nucleus, retain_dm=True,
                                          retain_conj=True, retain_adv=True,
                                          add_subject_to_rel_clause=True, resolve_coref=True)
    # full_satellite_pair = preprocessing_pipeline(context, satellite, retain_dm=True,
                                        #   retain_conj=True, retain_adv=True,
                                        #   add_subject_to_rel_clause=True, resolve_coref=True)
    # if cannot process one of the two
    if nucleus_pair[1] is None or satellite_pair[1] is None:
        return (False, ("", ""))

    # extract nucleus and satellite
    nucleus = nucleus_pair[1]
    satellite = satellite_pair[1]

    # if use template
    if use_template:
        if is_one_clause(nucleus, count_relative_clause=False):
            cl_type = check_relative_clause(nucleus)
            if cl_type != 0: # is relative clause
                rel_type = check_relative_clause_type(nucleus)
                if rel_type == 0:
                    ques, ans = enablement_question_type_0(nucleus, satellite)
                    return (True, (ques, ans))
                elif rel_type == 1:
                    ques, ans = enablement_question_type_1(nucleus, satellite)
                    return (True, (ques, ans))
            else:
                if is_dependent_clause(nucleus_pair[0], nucleus):
                    ques, ans = enablement_question_type_3(nucleus, satellite)
                    return (True, (ques, ans))
                ques, ans = enablement_question_type_2(nucleus, satellite)
                return (True, (ques, ans))
        return (False, ("", ""))

    # not use template -> use neural question generator
    else:
        # currently take full_answer (to feed to complete question model) to be the same as answer, \
        # which is basically the same as original nucleus, with no context added, \
        # might change later if needed.

        # full_answer = full_nucleus_pair[1].strip()
        last_noun_chunk = get_last_noun_chunk(full_nucleus_pair[1])
        main_verb = get_main_verb(full_nucleus_pair[1])
        key = "For what purpose"

        if last_noun_chunk is not None:
            key = key + ' ' + last_noun_chunk
        # if main_verb is not None:
        #     key = key + ' ' + main_verb

        full_answer = satellite.strip()
        question = complete_question(context, key, full_answer)
        answer = satellite.strip()

        question = postprocess_question(question)
        answer = postprocess_answer(answer)
        return (True, (question, answer))

# Manner-Means relation

# manner-means relation
# difference to manner-means: manner-means encapsulate manner-means, as all means can "enable" the goal,
# but manner-means also contains situational aid, an event lead (may not intentionally) to another event

# manner-means relation
def means_question_type_0(nucleus, satellite):
    """Make question based on manner-means relationship
    Type 0: satellite (result) is: relative clause: which + verb + clause (e.g. which made him happy.)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """
    # experimenting: exactly the same as type_1
    doc = nlp(nucleus)
    rel_pro = ""
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if "PRON" in token.pos_:
                rel_pro = token.text
                break

    assert len(rel_pro) > 0
    question = nucleus.replace(rel_pro.strip(), "What method ", 1) + '?'
    answer = satellite.strip() + '.'

    return (question, answer)

def means_question_type_1(nucleus, satellite):
    """Make question based on manner-means relationship
    Type 1: satellite (result) is: relative clause: which + verb + (not clause) (e.g. which caused the noise.)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """
    doc = nlp(nucleus)
    rel_pro = ""
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            if "PRON" in token.pos_:
                rel_pro = token.text
                break

    assert len(rel_pro) > 0
    question = nucleus.replace(rel_pro.strip(), "What method ", 1) + '?'
    answer = satellite.strip() + '.'

    return (question, answer)

def means_question_type_2(nucleus, satellite):
    """Make question based on manner-means relationship
    Type 2: satellite (result) is: full clause (not relative clause)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """

    if has_aux(nucleus):
        new_nuc = move_aux_to_beginning(nucleus)
    else:
        new_nuc = choose_and_replace_aux_for_verb(nucleus)
    question = "By what method " + new_nuc.strip() + '?'
    answer = satellite.strip() + '.'

    return (question, answer)

def means_question_type_3(nucleus, satellite):
    """Make question based on manner-means relationship
    Type 2: satellite (result) is: relative clause but not start with relative pronoun (most probably adverbial clause)

    Args:
        nucleus (str): nucleus stripped of ending punctuation and spaces
        satellite (str): satellite stripped of ending punctuation and spaces
    Returns:p
        tuple(ques, ans)
    """

    question = "What strategy can be employed " + nucleus.strip() + '?'
    answer = satellite.strip() + '.'

    return (question, answer)

def generate_means_question(context, nucleus, satellite, use_template=False):
    """Generate question based on Manner-Means relation.\
    Satellite (the means) is the answer.
    Nucleus (the end) forms the question.

    Args:
        context (str):
        nucleus (str):
        satellite (str):

    Returns:
        Tuple(Boolean, Tuple(str, str)): Boolean dictates if questions are plausible. \
            The second tuple is the question-answer pair.
    """
    nucleus_pair = preprocessing_pipeline(context, nucleus, add_subject_to_rel_clause=True,
                                          resolve_coref=True)
    satellite_pair = preprocessing_pipeline(context, satellite, retain_dm=True,
                                          retain_conj=True, retain_adv=True,
                                          add_subject_to_rel_clause=False, resolve_coref=False)
    # to feed to complete question model for full context
    full_nucleus_pair = preprocessing_pipeline(context, nucleus, retain_dm=True,
                                          retain_conj=True, retain_adv=True,
                                          add_subject_to_rel_clause=True, resolve_coref=True)
    # full_satellite_pair = preprocessing_pipeline(context, satellite, retain_dm=True,
                                        #   retain_conj=True, retain_adv=True,
                                        #   add_subject_to_rel_clause=True, resolve_coref=True)
    # if cannot process one of the two
    if nucleus_pair[1] is None or satellite_pair[1] is None:
        return (False, ("", ""))

    # extract nucleus and satellite
    nucleus = nucleus_pair[1]
    satellite = satellite_pair[1]

    # if use template
    if use_template:
        if is_one_clause(nucleus, count_relative_clause=False):
            cl_type = check_relative_clause(nucleus)
            if cl_type != 0: # is relative clause
                rel_type = check_relative_clause_type(nucleus)
                if rel_type == 0:
                    ques, ans =  means_question_type_0(nucleus, satellite)
                    return (True, (ques, ans))
                elif rel_type == 1:
                    ques, ans =  means_question_type_1(nucleus, satellite)
                    return (True, (ques, ans))
            else:
                if is_dependent_clause(nucleus_pair[0], nucleus):
                    ques, ans =  means_question_type_3(nucleus, satellite)
                    return (True, (ques, ans))
                ques, ans =  means_question_type_2(nucleus, satellite)
                return (True, (ques, ans))
        return (False, ("", ""))

    # not use template -> use neural question generator
    else:
        # currently take full_answer (to feed to complete question model) to be the same as answer, \
        # which is basically the same as original nucleus, with no context added, \
        # might change later if needed.

        # full_answer = full_nucleus_pair[1].strip()
        full_answer = satellite.strip()

        key = "How"
        last_noun_chunk = get_last_noun_chunk(full_nucleus_pair[1])
        if last_noun_chunk is not None:
            key = key + ' ' + last_noun_chunk

        question = complete_question(context, key, full_answer)
        answer = satellite.strip()

        question = postprocess_question(question)
        answer = postprocess_answer(answer)
        return (True, (question, answer))

# generate_means_question(sub_context, nuc, sate)
def generate_temporal_question(context, nucleus, satellite, use_template=False):
    """Generate a question based on the Temporal relation.
    Satellite (later event) is the answer.
    Nucleus (preceding event) forms the questiom.

    Args:
        context (str):
        nucleus (str):
        satellite (str):
        use_template (bool, optional): Defaults to False.
    """
    nucleus_pair = preprocessing_pipeline(context, nucleus, add_subject_to_rel_clause=True,
                                          resolve_coref=True)
    satellite_pair = preprocessing_pipeline(context, satellite,
                                            add_subject_to_rel_clause=False, resolve_coref=False)
    # to feed to complete question model for full context
    # full_nucleus_pair = preprocessing_pipeline(context, nucleus, retain_dm=True,
                                        #   retain_conj=True, retain_adv=True,
                                        #   add_subject_to_rel_clause=True, resolve_coref=True)
    # full_satellite_pair = preprocessing_pipeline(context, satellite, retain_dm=True,
                                        #   retain_conj=True, retain_adv=True,
                                        #   add_subject_to_rel_clause=True, resolve_coref=True)
    # if cannot process one of the two
    if nucleus_pair[1] is None or satellite_pair[1] is None:
        return (False, ("", ""))

    # extract nucleus and satellite
    nucleus = nucleus_pair[1]
    satellite = satellite_pair[1]

    # if use template
    if use_template:
        return (False, ("", ""))
    # not use template -> use neural question generator
    else:
        # currently take full_answer (to feed to complete question model) to be the same as answer, \
        # which is basically the same as original nucleus, with no context added, \
        # might change later if needed.

        # full_answer = full_nucleus_pair[1].strip()
        full_answer = satellite.strip()

        question = complete_question(context, "After", full_answer)
        answer = satellite.strip()

        question = postprocess_question(question)
        answer = postprocess_answer(answer)
        return (True, (question, answer))

# generate_temporal_question(sub_context, nuc, sate, use_template=False)
def generate_elaboration_question(context, nucleus, satellite, use_template=False):
    """Generate a question based on the Elaboration relation.
    Satellite (additional information) is the answer.
    Nucleus (basic information) forms the questiom.

    Args:
        context (str):
        nucleus (str):
        satellite (str):
        use_template (bool, optional): Defaults to False.
    """
    nucleus_pair = preprocessing_pipeline(context, nucleus, add_subject_to_rel_clause=True,
                                          resolve_coref=True)
    satellite_pair = preprocessing_pipeline(context, satellite,
                                            add_subject_to_rel_clause=False, resolve_coref=False)
    # to feed to complete question model for full context
    full_nucleus_pair = preprocessing_pipeline(context, nucleus, retain_dm=True,
                                          retain_conj=True, retain_adv=True,
                                          add_subject_to_rel_clause=True, resolve_coref=True)
    # full_satellite_pair = preprocessing_pipeline(context, satellite, retain_dm=True,
                                        #   retain_conj=True, retain_adv=True,
                                        #   add_subject_to_rel_clause=True, resolve_coref=True)
    # if cannot process one of the two
    if nucleus_pair[1] is None or satellite_pair[1] is None:
        return (False, ("", ""))

    # extract nucleus and satellite
    nucleus = nucleus_pair[1]
    satellite = satellite_pair[1]

    # if use template
    if use_template:
        return (False, ("", ""))
    # not use template -> use neural question generator
    else:
        # from observations, answer makes sense the most when is one sentence
        if not is_single_sentence(satellite.strip()):
            return (False, ("",""))
        # currently take full_answer (to feed to complete question model) to be the same as answer, \
        # which is basically the same as original nucleus, with no context added, \
        # might change later if needed.

        # full_answer = full_nucleus_pair[1].strip()
        full_answer = satellite.strip()

        key = "What"
        last_noun_chunk = get_last_noun_chunk(full_nucleus_pair[1])
        if last_noun_chunk is not None:
            key = key + ' ' + last_noun_chunk

        question = complete_question(context, key, full_answer) # don't change "What", need to be general like this
        answer = satellite.strip()

        question = postprocess_question(question)
        answer = postprocess_answer(answer)
        return (True, (question, answer))

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

LENGTH_UPPER_BOUND = 35
LENGTH_LOWER_BOUND = 2

def generate_cause_questions_from_discourse_analysis(context, questions_df, discourse_df):
    """Generate cause question and correponding distractors, and append it to the dataframe.

    Args:
        questions_df (DataFrame):
    """
    rel = "Cause"
    for trip in discourse_df[discourse_df['new_relation'] == rel].iterrows():
        nucleus = trip[1]['nucleus'].strip()
        satellite = trip[1]['satellite'].strip()
       
        if len(nucleus.split()) > LENGTH_UPPER_BOUND or len(satellite.split()) > LENGTH_UPPER_BOUND or len(nucleus.split()) < LENGTH_LOWER_BOUND or len(satellite.split()) < LENGTH_LOWER_BOUND:
            continue

        sub_context = get_subtext(context, nucleus, 500) # get 500 words

        flag, (ques, ans) = generate_cause_question(sub_context, nucleus, satellite)
        if not flag:
            continue
        
        datapoint = {'relation': [rel], 'nucleus': [nucleus], 'satellite': [satellite], 'question': [ques], 'answer': [ans]}
        questions_df = pd.concat([questions_df, pd.DataFrame(datapoint)], ignore_index=True)
    
    if discourse_df[discourse_df['new_relation'] == rel].shape[0] == 0:
        print(f"No nucleus-satellite pair found for relation {rel}\n")
    
    return questions_df

def generate_explanation_questions_from_discourse_analysis(context, questions_df, discourse_df):
    """Generate explanation question and correponding distractors, and append it to the dataframe.

    Args:
        questions_df (DataFrame):
    """
    rel = "Explanation"
    for trip in discourse_df[discourse_df['new_relation'] == rel].iterrows():
        nucleus = trip[1]['nucleus'].strip()
        satellite = trip[1]['satellite'].strip()
        
        if len(nucleus.split()) > LENGTH_UPPER_BOUND or len(satellite.split()) > LENGTH_UPPER_BOUND or len(nucleus.split()) < LENGTH_LOWER_BOUND or len(satellite.split()) < LENGTH_LOWER_BOUND:
            continue

        sub_context = get_subtext(context, nucleus, 500) # get 500 words
        flag, (ques, ans) = generate_cause_question(sub_context, satellite, nucleus)
        if not flag:
            continue
        
        datapoint = {'relation': [rel], 'nucleus': [nucleus], 'satellite': [satellite], 'question': [ques], 'answer': [ans]}
        questions_df = pd.concat([questions_df, pd.DataFrame(datapoint)], ignore_index=True)
    
    if discourse_df[discourse_df['new_relation'] == rel].shape[0] == 0:
        print(f"No nucleus-satellite pair found for relation {rel}\n")
    return questions_df
    
def generate_contrast_questions_from_discourse_analysis(context, questions_df, discourse_df):
    """Generate contrast question and correponding distractors, and append it to the dataframe.

    Args:
        questions_df (DataFrame):
    """
    rel = "Contrast"
    for trip in discourse_df[discourse_df['new_relation'] == rel].iterrows():
        nucleus = trip[1]['nucleus'].strip()
        satellite = trip[1]['satellite'].strip()
        
        if len(nucleus.split()) > LENGTH_UPPER_BOUND or len(satellite.split()) > LENGTH_UPPER_BOUND or len(nucleus.split()) < LENGTH_LOWER_BOUND or len(satellite.split()) < LENGTH_LOWER_BOUND:
            continue

        sub_context = get_subtext(context, nucleus, 500) # get 500 words
        flag, (ques, ans) = generate_contrast_question(sub_context, nucleus, satellite)
        if not flag:
            continue
        
        datapoint = {'relation': [rel], 'nucleus': [nucleus], 'satellite': [satellite], 'question': [ques], 'answer': [ans]}
        questions_df = pd.concat([questions_df, pd.DataFrame(datapoint)], ignore_index=True)
    
    if discourse_df[discourse_df['new_relation'] == rel].shape[0] == 0:
        print(f"No nucleus-satellite pair found for relation {rel}\n")
    return questions_df

def generate_condition_questions_from_discourse_analysis(context, questions_df, discourse_df):
    """Generate condition question and correponding distractors, and append it to the dataframe.

    Args:
        questions_df (DataFrame):
    """
    rel = "Condition"
    for trip in discourse_df[discourse_df['new_relation'] == rel].iterrows():
        nucleus = trip[1]['nucleus'].strip()
        satellite = trip[1]['satellite'].strip()
        
        if len(nucleus.split()) > LENGTH_UPPER_BOUND or len(satellite.split()) > LENGTH_UPPER_BOUND or len(nucleus.split()) < LENGTH_LOWER_BOUND or len(satellite.split()) < LENGTH_LOWER_BOUND:
            continue

        sub_context = get_subtext(context, nucleus, 500) # get 500 words
        flag, (ques, ans) = generate_condition_question(sub_context, nucleus, satellite)
        if not flag:
            continue
        
        datapoint = {'relation': [rel], 'nucleus': [nucleus], 'satellite': [satellite], 'question': [ques], 'answer': [ans]}
        questions_df = pd.concat([questions_df, pd.DataFrame(datapoint)], ignore_index=True)
    
    if discourse_df[discourse_df['new_relation'] == rel].shape[0] == 0:
        print(f"No nucleus-satellite pair found for relation {rel}\n")
    return questions_df
    
def generate_enablement_questions_from_discourse_analysis(context, questions_df, discourse_df):
    """Generate enablement question and correponding distractors, and append it to the dataframe.

    Args:
        questions_df (DataFrame):
    """
    rel = "Enablement"
    for trip in discourse_df[discourse_df['new_relation'] == rel].iterrows():
        nucleus = trip[1]['nucleus'].strip()
        satellite = trip[1]['satellite'].strip()
        
        if len(nucleus.split()) > LENGTH_UPPER_BOUND or len(satellite.split()) > LENGTH_UPPER_BOUND or len(nucleus.split()) < LENGTH_LOWER_BOUND or len(satellite.split()) < LENGTH_LOWER_BOUND:
            continue

        sub_context = get_subtext(context, nucleus, 500) # get 500 words
        flag, (ques, ans) = generate_enablement_question(sub_context, nucleus, satellite)
        if not flag:
            continue
        
        datapoint = {'relation': [rel], 'nucleus': [nucleus], 'satellite': [satellite], 'question': [ques], 'answer': [ans]}
        questions_df = pd.concat([questions_df, pd.DataFrame(datapoint)], ignore_index=True)
    
    if discourse_df[discourse_df['new_relation'] == rel].shape[0] == 0:
        print(f"No nucleus-satellite pair found for relation {rel}\n")
    return questions_df

def generate_means_questions_from_discourse_analysis(context, questions_df, discourse_df):
    """Generate means question and correponding distractors, and append it to the dataframe.

    Args:
        questions_df (DataFrame):
    """
    rel = "Manner-Means"
    for trip in discourse_df[discourse_df['new_relation'] == rel].iterrows():
        nucleus = trip[1]['nucleus'].strip()
        satellite = trip[1]['satellite'].strip()
        
        if len(nucleus.split()) > LENGTH_UPPER_BOUND or len(satellite.split()) > LENGTH_UPPER_BOUND or len(nucleus.split()) < LENGTH_LOWER_BOUND or len(satellite.split()) < LENGTH_LOWER_BOUND:
            continue

        sub_context = get_subtext(context, nucleus, 500) # get 500 words
        flag, (ques, ans) = generate_means_question(sub_context, nucleus, satellite)
        if not flag:
            continue

        datapoint = {'relation': [rel], 'nucleus': [nucleus], 'satellite': [satellite], 'question': [ques], 'answer': [ans]}
        questions_df = pd.concat([questions_df, pd.DataFrame(datapoint)], ignore_index=True)
    
    if discourse_df[discourse_df['new_relation'] == rel].shape[0] == 0:
        print(f"No nucleus-satellite pair found for relation {rel}\n")
    return questions_df

def generate_temporal_questions_from_discourse_analysis(context, questions_df, discourse_df):
    """Generate temporal question and correponding distractors, and append it to the dataframe.

    Args:
        questions_df (DataFrame):
    """
    rel = "Temporal"
    for trip in discourse_df[discourse_df['new_relation'] == rel].iterrows():
        nucleus = trip[1]['nucleus'].strip()
        satellite = trip[1]['satellite'].strip()
        
        if len(nucleus.split()) > LENGTH_UPPER_BOUND or len(satellite.split()) > LENGTH_UPPER_BOUND or len(nucleus.split()) < LENGTH_LOWER_BOUND or len(satellite.split()) < LENGTH_LOWER_BOUND:
            continue

        sub_context = get_subtext(context, nucleus, 500) # get 500 words
        flag, (ques, ans) = generate_temporal_question(sub_context, nucleus, satellite)
        if not flag:
            continue
        
        datapoint = {'relation': [rel], 'nucleus': [nucleus], 'satellite': [satellite], 'question': [ques], 'answer': [ans]}
        questions_df = pd.concat([questions_df, pd.DataFrame(datapoint)], ignore_index=True)
    
    if discourse_df[discourse_df['new_relation'] == rel].shape[0] == 0:
        print(f"No nucleus-satellite pair found for relation {rel}\n")
    return questions_df

def generate_elaboration_questions_from_discourse_analysis(context, questions_df, discourse_df):
    """Generate elaboration question and correponding distractors, and append it to the dataframe.

    Args:
        questions_df (DataFrame):
    """
    rel = "Elaboration"
    for trip in discourse_df[discourse_df['new_relation'] == rel].iterrows():
        nucleus = trip[1]['nucleus'].strip()
        satellite = trip[1]['satellite'].strip()
        
        if len(nucleus.split()) > LENGTH_UPPER_BOUND or len(satellite.split()) > LENGTH_UPPER_BOUND or len(nucleus.split()) < LENGTH_LOWER_BOUND or len(satellite.split()) < LENGTH_LOWER_BOUND:
            continue

        sub_context = get_subtext(context, nucleus, 500) # get 500 words
        flag, (ques, ans) = generate_elaboration_question(sub_context, nucleus, satellite)
        if not flag:
            continue

        datapoint = {'relation': [rel], 'nucleus': [nucleus], 'satellite': [satellite], 'question': [ques], 'answer': [ans]}
        questions_df = pd.concat([questions_df, pd.DataFrame(datapoint)], ignore_index=True)
    
    if discourse_df[discourse_df['new_relation'] == rel].shape[0] == 0:
        print(f"No nucleus-satellite pair found for relation {rel}\n")
    return questions_df
    
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

    edus = get_edus_from_file(edu_path)

    # get discourse analysis
    discourse_file_path = sample_path + article_name + "_discourse.csv"
    df = pd.read_csv(discourse_file_path)

    # assembly the whole piece of text from EDUs, to ensure allignment
    original_text = ""
    for edu in edus:
        original_text += edu.strip() + ' '

    sents = split_into_sentences(original_text) # text split into sentences
    paras = split_into_paras(raw_original_text, deli='\n\n')

    # get questions
    questions = pd.DataFrame(columns=['relation', 'nucleus', 'satellite', 'question', 'answer'])

    # full run of 8 relations: Cause, Explanation, Contrast, Condition, Enablement, 
    # Manner-Means, Temporal, Elaboration
    questions = generate_cause_questions_from_discourse_analysis(original_text, questions, df)
    questions = generate_explanation_questions_from_discourse_analysis(original_text, questions, df)
    questions = generate_contrast_questions_from_discourse_analysis(original_text, questions, df)
    questions = generate_condition_questions_from_discourse_analysis(original_text, questions, df)
    questions = generate_enablement_questions_from_discourse_analysis(original_text, questions, df)
    questions = generate_means_questions_from_discourse_analysis(original_text, questions, df)
    questions = generate_temporal_questions_from_discourse_analysis(original_text, questions, df)
    questions = generate_elaboration_questions_from_discourse_analysis(original_text, questions, df)

    questions_dest_path = sample_path + article_name + "_questions.csv"
    questions.to_csv(questions_dest_path, index=False) 
    print(f"Questions generated and saved successfully to {questions_dest_path}!")

if __name__ == '__main__':
    main()