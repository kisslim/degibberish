#%%
MODEL_NAME = "arnir0/Tiny-LLM"
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# %%

text = "暗黑魔法師".encode('big5').decode('gbk') # "穞堵臸猭畍"
#FAILED test_degibberish.py::test_score_encoding_pairs_full - AssertionError: Expected 'gbk' and 'big5' which has the loss 18.564227294921874, but got utf_16_be and cp437 which has the loss 15.534834289550782 instead.
text = text.encode('utf_16_be').decode('cp437')
tokenized_text = tokenizer(text, return_tensors='pt')
tokenized_text

# %%
given_conditions = tokenized_text['input_ids'][:, :-1]
given_attention_mask = tokenized_text['attention_mask'][:, :-1]
predict_labels = tokenized_text['input_ids'][:, 1:]

import torch
with torch.no_grad():
    model_outputs = model(given_conditions, attention_mask=given_attention_mask, labels=predict_labels)
#model_outputs = model(input_ids=given_conditions, attention_mask=given_attention_mask, labels=predict_labels)
model_outputs
# %%
float(model_outputs.loss) # default mean
# %%
model.loss_function
# %%
import torch
with torch.no_grad():
    model_outputs = model(given_conditions, attention_mask=given_attention_mask, labels=predict_labels, num_items_in_batch=1)
model_outputs
# %%
model_outputs.loss / len("穞堵臸猭畍")
# %%
text, tokenized_text, model_outputs
# %%
"穞堵臸猭畍".encode('utf_16_be').decode('cp437')
# %%
"穞堵臸猭畍".encode('utf_16_be').decode('gbk')

# %%
