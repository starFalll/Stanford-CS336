



import io
from typing import Iterable, Iterator
from multiprocessing import Pool
import numpy as np
import regex as re
import json
import pathlib
import warnings
from tqdm import tqdm
from tests.common import gpt2_bytes_to_unicode

# Optional C++ acceleration via cppyy
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"^cppyy(\.|$)")
        warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API", category=UserWarning)
        import cppyy  # type: ignore
    CPPYY_AVAILABLE = True
    print("CPP Available")
except Exception:
    cppyy = None  # type: ignore
    CPPYY_AVAILABLE = False

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MAX_PROCESSES_NUM = 1
# Compile pattern once for reuse
COMPILED_PAT = re.compile(PAT)

def split_into_parts(lst, n_parts=8):
    k, m = divmod(len(lst), n_parts)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n_parts)]


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = None
        self._special_pattern = None
        if special_tokens:
            # longest match
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            # Pre-compile special token pattern for fast matching
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            self._special_pattern = re.compile(pattern)
        
        self.vocab_to_index = {}
        for k, v in vocab.items():
            self.vocab_to_index[v] = k
        
        # Pre-compute merge rank for faster lookup
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}

        # Initialize C++ BPE merger if available
        self._use_cpp = False
        self._bpe_cpp = None
        if CPPYY_AVAILABLE:
            try:
                # ===== C++ side (replace your cppyy.cppdef string) =====
                cppyy.cppdef(
                r"""
                #include <vector>
                #include <string>
                #include <unordered_map>
                #include <limits>
                #include <utility>
                #include <thread>
                #include <algorithm>

                class BPE {
                public:
                    // Input: pieces (each piece is a raw-byte string via latin1)
                    // Output: merged tokens per piece
                    using out_type = std::vector<std::vector<std::string>>;

                    std::unordered_map<std::string, std::unordered_map<std::string, size_t>> merge_rank;

                    BPE(const std::vector<std::pair<std::string, std::string>>& merges) {
                        for (size_t i = 0; i < merges.size(); ++i) {
                            const auto& p = merges[i];
                            merge_rank[p.first][p.second] = i;
                        }
                    }

                    // Split raw-byte string into single-byte token strings
                    static std::vector<std::string> split_bytes(const std::string& s) {
                        std::vector<std::string> tokens;
                        tokens.reserve(s.size());
                        for (unsigned char c : s) {
                            tokens.emplace_back(1, static_cast<char>(c));
                        }
                        return tokens;
                    }

                    std::vector<std::string> merge_once(std::vector<std::string> tokens) const {
                        if (tokens.size() <= 1) return tokens;

                        while (true) {
                            size_t best_rank = std::numeric_limits<size_t>::max();
                            size_t best_pos  = tokens.size();
                            std::string best_a;
                            std::string best_b;

                            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                                auto it1 = merge_rank.find(tokens[i]);
                                if (it1 == merge_rank.end()) continue;
                                auto it2 = it1->second.find(tokens[i + 1]);
                                if (it2 == it1->second.end()) continue;
                                if (it2->second < best_rank) {
                                    best_rank = it2->second;
                                    best_pos = i;
                                    best_a = tokens[i];
                                    best_b = tokens[i + 1];
                                }
                            }

                            if (best_pos >= tokens.size()) break;

                            std::vector<std::string> new_tokens;
                            new_tokens.reserve(tokens.size());

                            size_t i = 0;
                            while (i < tokens.size()) {
                                if (i == best_pos && tokens[i] == best_a && tokens[i + 1] == best_b) {
                                    new_tokens.emplace_back(tokens[i] + tokens[i + 1]);
                                    i += 2;
                                } else {
                                    new_tokens.emplace_back(std::move(tokens[i]));
                                    i += 1;
                                }
                            }
                            tokens.swap(new_tokens);
                        }
                        return tokens;
                    }

                    void merge_job(
                        const std::vector<std::string>& pieces,
                        std::vector<out_type>& inter_vec,
                        int index,
                        int thread_num
                    ) const {
                        int size = (int)pieces.size();
                        int s = (size + thread_num - 1) / thread_num;
                        int begin = index * s;
                        if (begin >= size) return;
                        int end = std::min(begin + s, size);

                        inter_vec[index].reserve((size_t)(end - begin));
                        for (int i = begin; i < end; i++) {
                            auto tokens = split_bytes(pieces[i]);
                            inter_vec[index].emplace_back(merge_once(std::move(tokens)));
                        }
                    }

                    out_type merge(const std::vector<std::string>& pieces, int thread_num) const {
                        out_type out;
                        int size = (int)pieces.size();
                        if (size == 0) return out;
                        if (thread_num <= 0) thread_num = 1;
                        if (thread_num > size) thread_num = size;

                        out.reserve((size_t)size);
                        std::vector<out_type> inter_vec(thread_num);

                        std::vector<std::thread> threads;
                        threads.reserve(thread_num);
                        for (int i = 0; i < thread_num; i++) {
                            threads.emplace_back(&BPE::merge_job, this, std::ref(pieces), std::ref(inter_vec), i, thread_num);
                        }
                        for (auto& th : threads) th.join();

                        for (out_type& vec : inter_vec) {
                            out.insert(out.end(),
                                    std::make_move_iterator(vec.begin()),
                                    std::make_move_iterator(vec.end()));
                        }
                        return out;
                    }
                };
                """
                )


                from cppyy.gbl import std  # type: ignore
                self._std = std
                self._BPE_cpp = cppyy.gbl.BPE  # type: ignore

                merges_vec = std.vector[std.pair[std.string, std.string]]()
                for a, b in self.merges:
                    # Use latin1 to preserve raw byte values in a Python str
                    merges_vec.push_back(std.pair[std.string, std.string](a.decode('latin1'), b.decode('latin1')))

                self._bpe_cpp = self._BPE_cpp(merges_vec)
                self._use_cpp = True
            except Exception:
                self._bpe_cpp = None
                self._use_cpp = False

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

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        
        # Split text into chunks (special tokens or regular text)
        if self._special_pattern:
            chunks = self._special_pattern.split(text)
        else:
            chunks = [text]

        out_ids: list[int] = []
        pieces: list[bytes] = []

        def flush_pieces():
            nonlocal pieces, out_ids
            if not pieces:
                return
            out_ids.extend(self.merged_list(pieces))
            pieces.clear()
        
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                # special token should be a single token (no BPE merge)
                flush_pieces()
                b = chunk.encode("utf-8")
                out_ids.append(self.vocab_to_index[b])
                continue

            for match in COMPILED_PAT.finditer(chunk):
                bs = match.group().encode("utf-8")
                pieces.append(bs)

        flush_pieces()
        return out_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for elem in iterable:
            res = self.encode(elem)
            for id in res:
                yield id

    def decode(self, ids: list[int]) -> str:
        try:
            content_bytes = b''.join(self.vocab[id] for id in ids)
            return content_bytes.decode("utf-8", errors='replace')
        except Exception:
            return '\uFFFD'

    def merged_list(self, pieces: list[bytes]) -> list[int]:
        if not pieces:
            return []

        # Fast path: cppyy-backed merger (pieces -> C++ -> merged tokens)
        if getattr(self, "_use_cpp", False) and self._bpe_cpp is not None:
            try:
                std = self._std  # type: ignore[attr-defined]
                vec = std.vector[std.string]()

                for p in pieces:
                    vec.push_back(p.decode("latin1"))

                thread_num = min(MAX_PROCESSES_NUM, len(pieces))
                if len(pieces) < 5000:
                    thread_num = 1

                merged = self._bpe_cpp.merge(vec, thread_num)  # type: ignore[attr-defined]

                out: list[int] = []
                for i in range(merged.size()):
                    inner = merged[i]
                    for j in range(inner.size()):
                        py_s = str(inner[j])         
                        b = py_s.encode("latin1") 
                        out.append(self.vocab_to_index[b])

                return out
            except Exception:
                pass  # fallback

        # Python fallback (pieces -> byte tokens -> merge)
        # Convert each piece to list[bytes] of single bytes, then merge_once per piece
        encoding_results: list[int] = []

        def merge_once(tokens: list[bytes]) -> list[bytes]:
            if len(tokens) <= 1:
                return tokens
            merge_rank = self._merge_rank
            while True:
                best_rank = None
                best_pos = None
                best_a = best_b = None
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    r = merge_rank.get(pair)
                    if r is None:
                        continue
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best_pos = i
                        best_a, best_b = pair
                if best_rank is None:
                    break
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i == best_pos and tokens[i] == best_a and tokens[i + 1] == best_b:
                        new_tokens.append(best_a + best_b)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            return tokens

        for p in pieces:
            toks = [bytes([b]) for b in p]  # still slower, but fallback only
            toks = merge_once(toks)
            for t in toks:
                encoding_results.append(self.vocab_to_index[t])

        return encoding_results

        

if __name__ == "__main__":
    # TEST_PATH = (pathlib.Path(__file__).resolve().parent) / "../tests/fixtures"
    # VOCAB_PATH = TEST_PATH / "gpt2_vocab.json"
    # MERGES_PATH = TEST_PATH / "gpt2_merges.txt"
    # tokenizer = BPETokenizer.from_files(VOCAB_PATH, MERGES_PATH)
    # all_ids = []
    # with open(TEST_PATH / "tinystories_sample.txt") as f:
    #     for _id in tokenizer.encode_iterable(f):
    #         all_ids.append(_id)
    # with open(TEST_PATH / "tinystories_sample.txt") as f:
    #     corpus_contents = f.read()
    # print(all_ids)
    # print("-------------------")
    # res = tokenizer.decode(all_ids)
    # print(corpus_contents)
    # print("-------------")
    # print(res)
    # assert res == corpus_contents
    current_type = "train"
    VOCAB_PATH = f"{current_type}_vocab.json"
    MERGES_PATH = f"{current_type}_merges.txt"
    DATASET_PATH = pathlib.Path(__file__).resolve().parent.parent / f"data/TinyStoriesV2-GPT4-{current_type}.txt"
    BIN_OUT_PATH = f"GPT4-{current_type}.bin"

    tokenizer = BPETokenizer.from_files(VOCAB_PATH, MERGES_PATH)
    buffer_size = 10240
    buffer = np.empty(buffer_size, dtype=np.uint16)
    idx = 0
    total_written = 0
    batch_line = 10000

    # Count total lines efficiently
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Tokenize and write to binary file efficiently in buffered batches
    with io.open(BIN_OUT_PATH, "wb", buffering=io.DEFAULT_BUFFER_SIZE * 8) as output, \
         open(DATASET_PATH, "r", encoding="utf-8") as f:
        
        inter_lines = []
        line_num = 0
        processed = 0
        pbar = tqdm(f, total=total_lines, desc="Tokenizing & writing", unit=" lines")
        for line in pbar:
            inter_lines.append(line)
            line_num += 1
            processed += 1
            if line_num == batch_line or processed == total_lines:
                text = "".join(inter_lines)
                for token_id in tokenizer.encode(text):
                    buffer[idx] = token_id
                    idx += 1
                    if idx == buffer_size:
                        buffer.tofile(output)
                        total_written += idx
                        pbar.set_postfix_str(f"Written {total_written} tokens")
                        idx = 0
                inter_lines.clear()
                line_num = 0

        # Flush remainder
        if idx > 0:
            buffer[:idx].tofile(output)
            total_written += idx
            pbar.set_postfix_str(f"Written {total_written} tokens")
    