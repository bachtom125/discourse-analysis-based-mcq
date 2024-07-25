from edu_segmentation.download import download_models
from edu_segmentation.main import ModelFactory, EDUSegmentation
import argparse

download_models()

# Create a BERT Uncased model   
model = ModelFactory.create_model("bert_cased") # bart or bert_cased or bert_uncased

# Create an instance of EDUSegmentation using the model
edu_segmenter = EDUSegmentation(model)

# Segment the text using the conjunction-based segmentation strategy
granularity = "conjunction_words" # or default
conjunctions = [] # customise as needed e.g. ["and", "but"]
device = 'cpu' # or cuda

parser = argparse.ArgumentParser()
parser.add_argument('--input_data_file', default= '../data/sample/article')
parser.add_argument('--output_data_file', default= '../data/sample/article.edus')
args = parser.parse_args()

text = ""
with open(args.input_data_file, 'r', encoding='utf-8') as file:
    for line in file:
      if line.strip():
        text += line.strip()
      else:
        text += ' '

text = text.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\u2013", "-").replace(u"\u2014", "-").replace(u"\u201C", "-").replace(u"\u201D", "-")
segmented_output = edu_segmenter.run(text, granularity, conjunctions, device)

edus = ""
for edu_t in segmented_output:
  if not edu_t[1].strip():
    continue
  edus += edu_t[1].strip() + '\n'

with open (args.output_data_file, 'w') as f:
    f.write(edus)
print("\nSegmentation Finished!\n")
print("Written to ", args.output_data_file)