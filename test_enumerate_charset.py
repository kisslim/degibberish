import pytest
from enumerate_charset import find_common_decodable_bytes # Replace 'your_module' with the actual file name

def test_find_common_decodable_bytes_basic_case():
    """
    Tests a basic case with two known encodings that share ASCII characters.
    This test verifies that the generator produces the expected bytes.
    """
    # Arrange
    encodings = ('ascii', 'utf-8')
    max_length = 1  # Keep it short to ensure the test is fast

    # Act
    result = list(find_common_decodable_bytes(*encodings, max_length=max_length))

    # Assert
    # All ASCII characters from 0-127 should be decodable by both.
    expected_bytes = [bytes([i]) for i in range(128)]
    assert result == expected_bytes, "ASCII characters should be decodable by both 'ascii' and 'utf-8'"

@pytest.mark.parametrize(
    "encodings, max_length, expected_count",
    [
        # Test case: 'big5' and 'gb2312' share a lot of common single-byte characters.
        (('big5', 'gb2312'), 1, 128),
        # Test case: Different max_length
        (('ascii', 'utf-8'), 2, 128 + (128 * 128)),  # 1-byte + 2-byte ASCII
        # Test case: No common decodable bytes for a given max_length. iso-8855-1 seems to be a hallucination
        (('euc-jp', 'iso-8859-1'), 1, 128),
        # Test case: An encoding with no overlap with others for 1-byte.
        (('big5', 'ascii'), 1, 128),  # big5 shares ASCII range
    ]
)
def test_find_common_decodable_bytes_parameterized(encodings, max_length, expected_count):
    """
    Tests various encoding and length scenarios using parametrization.
    """
    # Arrange, Act, and Assert combined for clarity in this parameterized test
    results = list(find_common_decodable_bytes(*encodings, max_length=max_length))
    assert len(results) == expected_count, f"Expected {expected_count} decodable byte sequences for {encodings} with max_length={max_length}"

def test_find_common_decodable_bytes_empty_encodings():
    """
    Tests that if no encodings are provided, the generator yields all byte combinations.
    This covers an edge case in the function's logic.
    """
    # Arrange
    max_length = 1
    
    # Act
    result = list(find_common_decodable_bytes(max_length=max_length))
    
    # Assert
    expected_bytes = [bytes([i]) for i in range(256)]
    assert result == expected_bytes, "With no encodings, all bytes should be yielded."

def test_generator_exhaustion():
    """
    Tests that the generator is properly exhausted after yielding all results.
    """
    # Arrange
    gen = find_common_decodable_bytes('ascii', max_length=0)

    # Act & Assert
    with pytest.raises(StopIteration):
        next(gen)

# --- Code Coverage using pytest-cov ---
# To check for 100% coverage, we'll need to run this test with a coverage plugin.
# First, install it: `pip install pytest-cov`
# Then, add a .coveragerc file to configure what to measure (e.g., targeting the `your_module.py` file).
# `.coveragerc`
# [run]
# source = your_module.py
#
# Finally, run the tests with coverage flags: `pytest --cov=your_module --cov-report=html`
# This generates a report showing which lines of your code were executed.

