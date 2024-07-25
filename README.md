# Multiple Choice Question Generation using Discourse Analysis

This project focuses on automatically generating Multiple-Choice questions (MCQs) from text using Discourse Analysis. The pipeline involves segmenting and parsing the text, generating questions, and creating distractors. A combination of BERT, DeBERTa, T5 and other NLP techniques is used.

## Pipeline Overview

### Discourse Analysis

This component segments the text into Elementary Discourse Units (EDUs) and parses the relationships between them. Segbot is used for segmentation and Two-Stage Parser is used for parsing. An additional DeBERTa model is fine-tuned on relation labelling for improve accuracy.

### Question Generation

Based on the segmented and parsed text, this component generates questions. T5 fine-tuned on a combination of SQuAD, RACE, and NarrativeQA is used with a novel training strategy.

### Distractor Generation

For each generated question, this component creates plausible distractors. T5 fine-tuned on a combination of RACE, CosmosQA, and a Kaggle dataset is used with a custom loss function designed to punish repeating distractors.
