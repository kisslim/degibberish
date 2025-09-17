import os, codecs, encodings
import sys
from typing import Generator, List, TypedDict
# read environment variable
_debug = os.environ.get('DEBUG', False)

def log(*args, **kwargs):
    if _debug:
        print(*args, file=sys.stderr, flush=True, **kwargs)


def listcodecs(dir) -> List[str]:
    names = []
    for filename in os.listdir(dir):
        if filename[-3:] != '.py':
            continue
        name = filename[:-3]
        # Check whether we've found a true codec
        try:
            codecs.lookup(name)
        except LookupError:
            # Codec not found
            continue
        except Exception as reason:
            # Probably an error from importing the codec; still it's
            # a valid code name
            if _debug:
                print('* problem importing codec %r: %s' % \
                      (name, reason))
        names.append(name)
    return names

def get_encodings():
    return listcodecs(encodings.__path__[0])

import itertools
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# --- 1. Constants (not global variables) ---
# It's acceptable to define constants at the module level
# as they are not mutable during execution.
_DEFAULT_TEXT = "Hello, world! This is a test string with some special characters: é, ñ, ö, and 😊."
_DEFAULT_ENCODINGS = listcodecs(encodings.__path__[0])
_DEFAULT_MODEL_NAME = "fla-hub/rwkv7-0.1B-g1"

# --- 2. Reusable Functions ---

def load_model_and_tokenizer(model_name: str):
    """Loads a pre-trained language model and its tokenizer from Hugging Face."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load model '{model_name}': {e}")
        return None, None

def calculate_token_loss(new_text: str, tokenizer, model):
    """
    Calculates the average token loss of a decoded text relative to the original.
    """
    if not new_text:
        return float('inf')  # Infinite loss for an empty string
    with torch.no_grad():
        tokenized_text = tokenizer(new_text, return_tensors='pt')
        given_conditions = tokenized_text['input_ids'][:, :-1]
        given_attention_mask = tokenized_text['attention_mask'][:, :-1]
        predict_labels = tokenized_text['input_ids'][:, 1:]
        model_outputs = model(given_conditions, attention_mask=given_attention_mask, labels=predict_labels)
        return float(model_outputs.loss) # default mean


# a TypedDict for the results

class EncodingPairScoringResult(TypedDict):
    encode_A: str
    decode_B: str
    avg_token_loss: float
    decoded_text: str | None
    error: str | None

    
class Result:
    def __init__(self, encode_A, decode_B, avg_token_loss, decoded_text, error=None):
        self.encode_A = encode_A
        self.decode_B = decode_B
        self.avg_token_loss = avg_token_loss
        self.decoded_text = decoded_text
        self.error = error

def score_encoding_pairs(text: str, model_name: str | None = None, model: torch.nn.Module | AutoModelForCausalLM | None = None, tokenizer: AutoTokenizer | None = None, encodings: list | None = None) -> Generator[EncodingPairScoringResult, None, None]:
    """
    Enumerates all encoding/decoding pairs and scores them using a language model.
    """
    trust_remote_code = False
    if model_name is None:
        model_name = _DEFAULT_MODEL_NAME
        trust_remote_code = True
    if encodings is None:
        encodings = _DEFAULT_ENCODINGS
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    for A, B in itertools.product(encodings, encodings):
        decoded_text = None
        loss = float('inf')
        error = None

        try:
            # Attempt to encode and decode
            encoded_bytes = text.encode(A)
            decoded_text = encoded_bytes.decode(B)
            
            # Calculate the token loss
            loss = calculate_token_loss(decoded_text, tokenizer, model)

        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            error = str(e)
        except Exception as e:
            error = f"Unexpected error: {e}"

        yield {
            'encode_A': A,
            'decode_B': B,
            'decoded_text': decoded_text,
            'avg_token_loss': loss,
            'error': error
        }
        
        # Optional: Print progress
        if error:
            print(f"Encode with '{A}', Decode with '{B}': Error - {error}")
        else:
            print(f"Encode with '{A}', Decode with '{B}': Loss = {loss:.4f}")


def print_results(results):
    """Prints the sorted results in a user-friendly format."""
    log("\n--- Summary of Results (Best to Worst) ---")
    for res in sorted(results, key=lambda x: x['avg_token_loss']):
        loss = f"{res['avg_token_loss']:.4f}" if res['avg_token_loss'] != float('inf') else "Error"
        decoded_preview = res['decoded_text'][:40] + "..." if res['decoded_text'] and len(res['decoded_text']) > 40 else res['decoded_text']
        
        log(f"Encode: {res['encode_A']:<10} | Decode: {res['decode_B']:<10} | Avg Token Loss: {loss:<10} | Decoded: {decoded_preview if decoded_preview else 'N/A'} | Error: {res['error'] if res['error'] else 'N/A'}")

# --- 3. Main Function ---

def main():
    """
    Main function to run the encoding scoring script.
    Encapsulates the primary logic for a direct execution.
    """
    tokenizer, model = load_model_and_tokenizer(_DEFAULT_MODEL_NAME)
    
    if not all([tokenizer, model]):
        print("Exiting due to model loading failure.")
        return
        
    results = score_encoding_pairs(_DEFAULT_TEXT, encodings=_DEFAULT_ENCODINGS, model=model, tokenizer=tokenizer)
    print_results(results)

# --- 4. Entry Point ---

if __name__ == "__main__":
    main()

