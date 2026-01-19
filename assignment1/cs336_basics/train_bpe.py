import os
import pathlib
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
from pprint import pprint
import traceback
import json
from tests.common import gpt2_bytes_to_unicode

DEBUG_PRE_TOKEN_FILE="pre_token.json"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PROCESSES_NUM = 8

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_tokens, list) and all(isinstance(x, bytes) for x in split_special_tokens), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            
            found_at = -1
            # Find the special token in the mini chunk
            for split_special_token in split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
            if found_at != -1:
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_token_process(corpus: str, special_tokens: list) -> dict[tuple[bytes], int]:
    pattern = "|".join(re.escape(t) for t in special_tokens)
    chunks = re.split(f"{pattern}", corpus)
    kv = {}
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            res = tuple(bytes([b]) for c in match.group() for b in c.encode("utf-8"))
            if res not in kv:
                kv[res] = 1
            else:
                kv[res] += 1
    return kv
        

def pre_tokenization(file: BinaryIO, num_processes: int, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    pre_token_results = {}
    with open(file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, [token.encode("utf-8") for token in special_tokens])

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # print("len:", len(boundaries))
        # print(boundaries)
        pre_token_parameters_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pre_token_parameters_list.append([chunk, special_tokens])
            #test
            # print("current", len(pre_token_parameters_list))
            # with open(f"chunk_{len(pre_token_parameters_list)}.txt", "w") as f2:
            #     f2.write(chunk)
            # Run pre-tokenization on your chunk and store the counts for each pre-token
        with Pool() as pool:
            async_results = [pool.apply_async(pre_token_process, param) for param in pre_token_parameters_list]
            for r in async_results:
                for k, v in r.get().items():
                    pre_token_results[k] = pre_token_results.get(k, 0) + v

    # with open(DEBUG_PRE_TOKEN_FILE, "w") as f:
    #     sorted_items = sorted(pre_token_results.items(), key=lambda kv: kv[1], reverse=True)
    #     for k, v in sorted_items:
    #         f.write(f"{k}: {v}\n")
    return pre_token_results


def update_merged_dict(merged_dict, new_key, value, index):
    if new_key == None:
        return
    if new_key not in merged_dict:
        merged_dict[new_key] = [value, [index]]
    else:
        merged_dict[new_key][0] += value
        merged_dict[new_key][1].append(index)

def delete_old_pairs(merged_dict, old_key, value):
    merged_dict[old_key][0] -= value
    if merged_dict[old_key][0] == 0:
        del merged_dict[old_key]

def token_merge(
    pre_token_results: dict[tuple[bytes], int], 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # skip all single byte

    # build indexes
    index_to_dictkey = {}
    dictkey_to_index = {} 
    for i, k in enumerate(pre_token_results.keys()):
        if len(k) == 1:
            continue
        index_to_dictkey[i] = k
        dictkey_to_index[k] = i
    # tuple(token1, token2)->(value, [index1, index2])
    merged_dict : dict[tuple[bytes, bytes], list] = {}
    vocab : dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    vocab.update({i: special_tokens[i-256].encode("utf-8") for i in range(256, len(special_tokens)+256)})
    cur_vocab_index = len(vocab)
    merges : list[tuple[bytes, bytes]] = []
    finished_merges = 0

    # initialize first merge
    for key, value in pre_token_results.items():
        if len(key) > 1:
            for i in range(len(key)-1):
                new_key = (key[i],key[i+1])
                update_merged_dict(merged_dict, new_key, value, dictkey_to_index[key])
        # else:
        #     update_merged_dict(merged_dict, (key), value, dictkey_to_index)
    
    # print("First merge")
    # pprint(merged_dict)
    # print("End first merge")
    # with open("test_merge.txt", "r+") as f:
    #         f.truncate(0)
    while len(vocab) < vocab_size and finished_merges < len(dictkey_to_index):
        max_item = max(merged_dict.items(), key=lambda kv: (kv[1][0], kv[0]))
        tokens = max_item[0]
        # test
        count = {}
        for k, v in merged_dict.items():
            if v[0] == max_item[1][0]:
                count[k] = v[0]
        # with open("test_merge.txt", "a") as f:
        #     f.write(f"token:{tokens}")
        #     f.write(repr(count)+"\n")
        #test
        
        new_merged_token = tokens[0]+tokens[1]
        # print("new round", tokens, new_merged_token)
        used_index = set()
        for index in max_item[1][1]:
            if index in used_index:
                continue
            used_index.add(index)
            key = index_to_dictkey[index]
            new_pre_token_list = []
            value = pre_token_results[key]
            i = 0

            while i < len(key):
                if i < len(key) - 1 and key[i] == tokens[0] and key[i+1] == tokens[1]:
                    
                    if new_pre_token_list:
                        old_key = (key[i-1], key[i])
                        # use new pairs as new key
                        new_key = (new_pre_token_list[-1], new_merged_token)
                        delete_old_pairs(merged_dict, old_key, value) 
                        update_merged_dict(merged_dict, new_key, value, index)
                    delete_old_pairs(merged_dict, tokens, value)
                    # if tokens are not repeated or it is last element, count
                    if i == len(key) - 3 or (i < len(key)-3 and (key[i+2] != tokens[0] or key[i+3] != tokens[1])):
                        old_key = (key[i+1], key[i+2])
                        new_key = (new_merged_token, key[i+2])
                        delete_old_pairs(merged_dict, old_key, value) 
                        update_merged_dict(merged_dict, new_key, value, index) 
                    
                    
                    new_pre_token_list.append(new_merged_token)
                    i += 2
                else:
                    new_pre_token_list.append(key[i])
                    i += 1
            new_pre_token = tuple(new_pre_token_list)
            # print(new_pre_token)
            # pprint(merged_dict)
            if len(new_pre_token) == 1:
                finished_merges += 1
            del dictkey_to_index[key] 
            del pre_token_results[key]
            pre_token_results[new_pre_token] = value
            dictkey_to_index[new_pre_token] = index
            index_to_dictkey[index] = new_pre_token

        # print("tokens:", tokens)
        merges.append(tokens)
        
        vocab[cur_vocab_index]=new_merged_token
        cur_vocab_index += 1
        if tokens in merged_dict:
            del merged_dict[tokens]
    
    return [vocab, merges]




def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    processes_num: int = PROCESSES_NUM
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    try:
        pre_token_results = pre_tokenization(input_path, processes_num, special_tokens)
        return token_merge(pre_token_results, vocab_size, special_tokens)
    except Exception as e:
        print(f"An unexpected error occurred: {e}, trace:{traceback.format_exc()}")

def bytes_to_unicode_escape(b: bytes) -> str:
    print(b)
    return ''.join(f'\\u{byte:04X}' for byte in b)

def train_bpe(
    vocab_file,
    merges_file,
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    processes_num: int = PROCESSES_NUM):
    [vocab, merges] = run_train_bpe(input_path, vocab_size, special_tokens, processes_num)
    # Get GPT-2 byte → unicode mapping
    gpt2_byte_decoder = gpt2_bytes_to_unicode()  # dict[int, str]

    # Convert your vocab bytes to unicode strings using GPT-2 mapping
    vocab_to_save = {
        ''.join(gpt2_byte_decoder[b] for b in token): idx
        for idx, token in vocab.items()
    }
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_to_save, f, indent=2, ensure_ascii=False)
    # with open(merges_file, "w", encoding="utf-8") as f:
    #     for merge_token_1, merge_token_2 in merges:
    #         f.write(f"{merge_token_1} {merge_token_2}\n")
    with open(merges_file, "w", encoding="utf-8") as f:
        for merge_token_1, merge_token_2 in merges:
            # Convert each byte in the tuple to a printable string
            token1 = ''.join(gpt2_byte_decoder[b] for b in merge_token_1)
            token2 = ''.join(gpt2_byte_decoder[b] for b in merge_token_2)
            f.write(f"{token1} {token2}\n")

if __name__ == "__main__": 
    SPECIAL_TOKENS = "<|endoftext|>"
    valid_path = pathlib.Path(__file__).resolve().parent.parent / "data/TinyStoriesV2-GPT4-train.txt"
    train_bpe("train_vocab.json", "train_merges.txt", valid_path, 10000, [SPECIAL_TOKENS])
    # [vocab, merges] = run_train_bpe(valid_path, 10000, [SPECIAL_TOKENS], 8) 
    # count = 0
    # vocab_str = {}
    # for k, v in vocab.items():
    #     vocab_str[str(v)] = k 
    # #    vocab_str = {v.decode("utf-8"): k for k, v in vocab.items()}
    # with open("valid_vocab.json", "w", encoding="utf-8") as f:
    #     json.dump(vocab_str, f, ensure_ascii=False)
        
    # with open("valid_merges.txt", "w") as f:
    #     for pair in merges:
    #         # f.write(" ".join(b.decode("utf-8", errors="replace") for b in pair))
    #         f.write(" ".join(b for b in pair))
    #         f.write("\n")
    # SPECIAL_TOKENS = "<|endoftext|>"
    # valid_path = pathlib.Path(__file__).resolve().parent.parent / "tests/fixtures/corpus.en"
    # vocab, merges = run_train_bpe(valid_path, 500, [SPECIAL_TOKENS], 8)
    # print(vocab, merges) 