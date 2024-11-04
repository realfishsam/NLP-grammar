import csv
import logging
import os

from nltk.tokenize.treebank import TreebankWordDetokenizer


SOURCE_EXTENSION = ".src"
TARGET_EXTENSION = ".tgt"

MAX_INPUT_LENGTH = 3000
MAX_OUTPUT_LENGTH = 3000

MODEL_SAVE_DIR = "fine_tuned_model/"

DETOKENIZER = TreebankWordDetokenizer()


def detokenize(text):
    detokenized = DETOKENIZER.detokenize(text.split())
    return detokenized


def clear_all_csv_files(file_dirs):
    """
    Clear all csv files in the data folder
    """
    for file_dir in file_dirs:
        for file in os.listdir(file_dir):
            if file.endswith(".csv"):
                os.remove(os.path.join(file_dir, file))


def load_dataset(file_path, file_name_prefix, has_label=True, detokenize_input=True):
    """
    Load dataset from file and write to a csv file
    """
    csv_path = os.path.join(file_path, file_name_prefix + ".csv")
    dataset_source_path = os.path.join(file_path, file_name_prefix + SOURCE_EXTENSION)
    dataset_target_path = os.path.join(file_path, file_name_prefix + TARGET_EXTENSION)

    max_source_length = 0
    max_target_length = 0

    # check if csv file already exists
    if os.path.exists(csv_path):
        logging.info(f"File {csv_path} already exists")
        return csv_path

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if has_label:
            writer.writerow(["input", "target"])
            with open(dataset_source_path) as src_file:
                with open(dataset_target_path) as trg_file:
                    for src, trg in zip(src_file, trg_file):
                        if src.strip() == "" or trg.strip() == "":
                            continue

                        max_source_length = max(max_source_length, len(src))
                        max_target_length = max(max_target_length, len(trg))

                        src_text = src.strip()
                        src_text = (
                            detokenize(src_text) if detokenize_input else src_text
                        )
                        src_text = f"grammar: {src_text}"

                        trg_text = trg.strip()
                        trg_text = (
                            detokenize(trg_text) if detokenize_input else trg_text
                        )

                        writer.writerow([src_text, trg_text])
        else:
            writer.writerow(["input"])
            with open(dataset_source_path) as src_file:
                for src in src_file:
                    if src.strip() == "":
                        continue

                    max_source_length = max(max_source_length, len(src))

                    src_text = src.strip()
                    src_text = detokenize(src_text) if detokenize_input else src_text
                    src_text = f"grammar: {src_text}"

                    writer.writerow([src_text])
    logging.info(f"File {csv_path} created")
    # logging.info(f"Max source length: {max_source_length}")
    # logging.info(f"Max target length: {max_target_length}")
    return csv_path


def load_datasets(
    file_path, file_name_prefixes, has_labels=None, detokenize_input=True
):
    """
    Load multiple datasets from files and write to csv files
    """
    if has_labels is None:
        has_labels = [True] * len(file_name_prefixes)

    csv_paths = []
    for file_name_prefix, has_label in zip(file_name_prefixes, has_labels):
        csv_paths.append(
            load_dataset(file_path, file_name_prefix, has_label, detokenize_input)
        )
    return csv_paths


logging.basicConfig(
    filename=f"output.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)
