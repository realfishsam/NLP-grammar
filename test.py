import csv
import os

from happytransformer import HappyTextToText, TTSettings
import spacy

from common import (
    detokenize,
    MODEL_SAVE_DIR,
    MAX_OUTPUT_LENGTH,
)

TEST_DATASETS_FILE_PATH = os.path.join(os.getcwd(), "data", "test")

TEST_BEA_2019_FILE_NAME = "ABCN.test.bea19.orig"

TOKENIZER = spacy.load("en_core_web_sm")


def tokenize(text):
    tokenized = " ".join([token.text for token in TOKENIZER(text)])
    return tokenized


def get_bea19_test_dataset():
    data = []
    with open(
        os.path.join(TEST_DATASETS_FILE_PATH, TEST_BEA_2019_FILE_NAME)
    ) as src_file:
        for src in src_file:
            if src.strip() == "":
                continue

            original_src = src.strip()
            src_text = detokenize(original_src)
            src_text = f"grammar: {src_text}"
            data.append((src_text, original_src))

    return data


def correct_grammar(model, data):
    args = TTSettings(
        num_beams=5, min_length=1, max_length=MAX_OUTPUT_LENGTH, early_stopping=True
    )
    corrected_data = []
    for src_text, original_src in data:
        result = model.generate_text(src_text, args=args)
        result_text = tokenize(result.text)
        corrected_data.append((original_src, result_text))
    return corrected_data


model = HappyTextToText("t5", model_name=MODEL_SAVE_DIR)

data = get_bea19_test_dataset()

corrected_data = correct_grammar(model, data)

with open("bea2019", "w") as f:
    for original, corrected in corrected_data:
        f.write(f"{corrected}\n")