



from typing import Iterable, Iterator
from multiprocessing import Pool
import regex as re
import json
import pathlib
from functools import lru_cache

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MAX_PROCESSES_NUM = 8

def split_into_parts(lst, n_parts=8):
    k, m = divmod(len(lst), n_parts)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n_parts)]


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = None
        if special_tokens:
            # longest match
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.vocab_to_index = {}
        for k, v in vocab.items():
            self.vocab_to_index[v] = k


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)
        # merges = []
        # with open(vocab_filepath, "r", encoding="utf-8") as f:
        #     vocab_str = json.load(f)
        # vocab = {int(v): k.encode("utf-8") for k, v in vocab_str.items()}
        # #test
        # count = 1
        # with open(merges_filepath, "r", encoding="utf-8") as f:
        #     for line in f:
        #         line = line.rstrip("\n")
        #         if not line:
        #             continue
        #         i = line.rfind(" ")
        #         if i == -1:
        #             raise ValueError(f"Invalid line (no separator): {line}, {count}")
        #         token1 = line[:i].encode("utf-8") if line[:i] else b" "
        #         token2 = line[i+1:].encode("utf-8") if line[i+1:] else b" "
        #         merges.append((token1, token2))
        #         count += 1
        # return cls(vocab, merges, special_tokens) 

    def encode(self, text: str) ->list[int]:
        # pattern = "|".join(re.escape(t) for t in self.special_tokens)
        # Build regex pattern, wrap in parentheses to capture
        if not text:
            return []
        # print(self.special_tokens)
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            # print(pattern)
            chunks = re.split(f"{pattern}", text)
        else:
            chunks = [text]
        encoding_list = []
        # print("chunks", chunks)
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                encoding_list.append([chunk.encode("utf-8")])
                continue
            for match in re.finditer(PAT, chunk):
                encoding_list.append(list(bytes([b]) for c in match.group() for b in c.encode("utf-8")))
        # print(encoding_list)
        split_list = split_into_parts(encoding_list, min(len(encoding_list), MAX_PROCESSES_NUM))
        results = []
        # print("split:", len(split_list), split_list)
        # print(split_list[0])
        with Pool() as pool:
            async_results = [pool.apply_async(self.merged_list, args=(param,)) for param in split_list]
            for r in async_results:
                results.extend(r.get())
        # for param in split_list:
        #     results.extend(self.merged_list(param))
        
        return results
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for elem in iterable:
            res = self.encode(elem)
            for id in res:
                yield id

    def decode(self, ids: list[int]) -> str:
        try:
            content_bytes = b''
            for id in ids:
                content_bytes += self.vocab[id]
            return content_bytes.decode("utf-8", errors='replace')
        except Exception:
            return '\uFFFD'


    
    def merged_list(self, encoding_list: list[list[bytes]]) -> list[int]:
        # print("before---------")
        # pprint(encoding_list)
        for pair in self.merges:
            # print("pair", pair)
            for idx, encoding in enumerate(encoding_list):
                i = 0
                new_encoding = []
                while i < len(encoding):
                    if i + 1 < len(encoding) and encoding[i] == pair[0] and encoding[i+1] == pair[1]:
                        new_encoding.append(pair[0]+pair[1])
                        i += 2
                    else:
                        new_encoding.append(encoding[i])
                        # print("current:", encoding[i])
                        i += 1
                encoding_list[idx] = new_encoding
                new_encoding = []
        encoding_results = []
        # print("after---------")
        # pprint(encoding_list)
        for encoding in encoding_list:
            for byte in encoding:
                try:
                    if isinstance(byte, bytes):
                        # print(byte)
                        encoding_results.append(self.vocab_to_index[byte])
                    else:
                        encoding_results.append(self.vocab_to_index[bytes([int(byte)])])
                except Exception as e:
                    print("error:", byte, type(byte), e)

        return encoding_results
        

if __name__ == "__main__":
    TEST_PATH = (pathlib.Path(__file__).resolve().parent) / "../tests/fixtures"
    VOCAB_PATH = TEST_PATH / "gpt2_vocab.json"
    MERGES_PATH = TEST_PATH / "gpt2_merges.txt"
    tokenizer = BPETokenizer.from_files(VOCAB_PATH, MERGES_PATH)
    all_ids = []
    with open(TEST_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    with open(TEST_PATH / "tinystories_sample.txt") as f:
        corpus_contents = f.read()
    print(all_ids)
    print("-------------------")
    res = tokenizer.decode(all_ids)
    print(corpus_contents)
    print("-------------")
    print(res)
    assert res == corpus_contents