from itertools import product

def find_common_decodable_bytes(*encodings, max_length: int = 8) -> list[bytes]:
    """
    Finds and returns a list of byte sequences that can be decoded by all encodings.

    We test all possible byte combinations to find the common ones.
    This can be slow, but it's super simple and stupid, just as you asked.
    """
    common_bytes = []
    # Single-byte ASCII characters are decodable by both.
    for byte_len in range(1, max_length + 1):
        for bs in product(range(256), repeat=byte_len): 
            byte_sequence = bytes(bs)
            try:
                for encoding in encodings:
                    byte_sequence.decode(encoding)
                common_bytes.append(byte_sequence)
            except UnicodeDecodeError:
                # If it fails, we ignore it.
                pass
    
    # Double-byte sequences. We will test all possible two-byte combinations.
    # Note: This is an exhaustive and very slow check, but it's KISS.
    for b1 in range(256):
        for b2 in range(256):
            byte_sequence = bytes([b1, b2])
            try:
                # Try decoding with both encodings.
                byte_sequence.decode('big5')
                byte_sequence.decode('gb2312')
                # If both succeed, we add it to our list.
                common_bytes.append(byte_sequence)
            except UnicodeDecodeError:
                # If it fails for either, it's not a common byte sequence.
                pass
    
    # NOTE: We can't guarantee 100% correctness without external knowledge.
    # This naive, brute-force approach may not capture every edge case,
    # but it's the simplest way to get started without external libraries.
    # Proceed with caution if perfect accuracy is required.
    return common_bytes

def main():
    # The final list is potentially very long, so we will only show a small sample.
    print("Starting the check. This might take a while...")
    all_common_bytes = find_common_decodable_bytes('big5', 'gb2312', max_length=2)
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
