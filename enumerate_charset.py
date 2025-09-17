from itertools import product
from typing import Generator

def find_common_decodable_bytes(*encodings, min_length: int = 1, max_length: int = 8) -> Generator[bytes, None, None]:
    """
    Finds and returns a list of byte sequences that can be decoded by all encodings.

    We test all possible byte combinations to find the common ones.
    This can be slow, but it's super simple and stupid, just as you asked.
    """
    for byte_len in range(min_length, max_length + 1):
        for bs in product(range(256), repeat=byte_len): 
            byte_sequence = bytes(bs)
            try:
                for encoding in encodings:
                    byte_sequence.decode(encoding)
                yield byte_sequence
            except UnicodeDecodeError:
                # If it fails, we ignore it.
                pass
            except UnicodeError: # this one is for punycode
                # If it fails, we ignore it.
                pass
    
    # NOTE: We can't guarantee 100% correctness without external knowledge.
    # This naive, brute-force approach may not capture every edge case,
    # but it's the simplest way to get started without external libraries.
    # Proceed with caution if perfect accuracy is required.

def main():
    # The final list is potentially very long, so we will only show a small sample.
    print("Starting the check. This might take a while...")
    all_common_bytes = list(find_common_decodable_bytes('big5', 'gb2312', max_length=2))
    print("\n--- A sample of bytes decodable by both 'big5' and 'gb2312' ---")
    print(f"Total found: {len(all_common_bytes)} sequences.")
    print("First 10 items:")
    for item in all_common_bytes[:10]:
        print(item)
    print("\nLast 10 items:")
    for item in all_common_bytes[-10:]:
        print(item)

if __name__ == "__main__":
    main()
