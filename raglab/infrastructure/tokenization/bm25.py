from bm25s.tokenization import Tokenizer


def _tokenize(text: str) -> list[str]:
    """Tokenize text using lowercasing and whitespace splitting"""

    return text.lower().split()


def build_tokenizer() -> Tokenizer:
    """Create the tokenizer shared by indexing and retrieval"""

    return Tokenizer(
        splitter=_tokenize,
        stopwords=[],
    )
