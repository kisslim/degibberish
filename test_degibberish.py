#pytest for degibberish module 
from itertools import combinations

from degibberish import score_encoding_pairs
from degibberish import get_encodings
from degibberish import print_results


from transformers import AutoModelForCausalLM, AutoTokenizer
import pytest
import torch

MODEL_NAME = "arnir0/Tiny-LLM"

@pytest.fixture()
def model(request):
    print("setup model", MODEL_NAME)

    def teardown():
        print("teardown")
    request.addfinalizer(teardown)
    
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

@pytest.fixture()
def tokenizer(request):
    print("setup tokenizer", MODEL_NAME)
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

class TestResource:
    def test_llm_tokenizer(self, tokenizer: AutoTokenizer):
        with torch.no_grad():
            new_tokens = tokenizer("暗黑魔法師".encode('big5').decode('gbk') , return_tensors='pt')
            assert isinstance(new_tokens['input_ids'], torch.Tensor)
        
    def test_llm_logits(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
        with torch.no_grad():
            new_tokens = tokenizer("暗黑魔法師".encode('big5').decode('gbk') , return_tensors='pt')
            new_outputs = model(**new_tokens)
            assert isinstance(new_outputs['logits'], torch.Tensor)

class TestDegibberish:
    def test_score_encoding_pairs(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        text = "暗黑魔法師".encode('big5').decode('gbk') # "穞堵臸猭畍"
        encodings = ['gbk', 'big5']
        results = list(score_encoding_pairs(text, encodings=encodings, model=model, tokenizer=tokenizer))
        assert len(results) == len(encodings) ** 2
    
    def test_score_encoding_pairs_(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        text = "暗黑魔法師".encode('big5').decode('gbk') # "穞堵臸猭畍"
        encodings = ['gbk', 'big5']
        results = list(score_encoding_pairs(text, encodings=encodings, model=model, tokenizer=tokenizer))
        assert len(results) == len(encodings) ** 2
        # should be gbk then big5
        min_loss_item = min(results, key=lambda x: x['avg_token_loss'])
        assert min_loss_item['encode_A'] == 'gbk'
        assert min_loss_item['decode_B'] == 'big5'

# FAILED test_degibberish.py::TestDegibberish::test_score_encoding_pairs_full - AssertionError: Expected 'gbk' and 'big5' which has the loss 18.564227294921874, but got utf_16_be and cp437 which has the loss 15.534834289550782 instead.
@pytest.mark.slow
def test_score_encoding_pairs_full():
    text = "暗黑魔法師".encode('big5').decode('gbk') # "穞堵臸猭畍"
    results = list(score_encoding_pairs(text))
    expected_loss_item = None
    for item in results:
        if item['encode_A'] == 'gbk' and item['decode_B'] == 'big5':
            expected_loss_item = item
    assert expected_loss_item is not None, "Expected 'gbk' and 'big5', but could not find it in the results."
    # should be gbk then big5
    min_loss_item = min(results, key=lambda x: x['avg_token_loss'])
    assert min_loss_item['encode_A'] == 'gbk' and min_loss_item['decode_B'] == 'big5', f"Expected 'gbk' and 'big5' which has the loss {expected_loss_item['avg_token_loss']}, but got {min_loss_item['encode_A']} and {min_loss_item['decode_B']} which has the loss {min_loss_item['avg_token_loss']} instead."

# still buggy. logs:

# Encode: gbk        | Decode: gbk        | Avg Token Loss: 8.7932     | Decoded: 穞堵臸猭畍 | Error: N/A
# Encode: gbk        | Decode: big5       | Avg Token Loss: 10.3135    | Decoded: 暗黑魔法師 | Error: N/A