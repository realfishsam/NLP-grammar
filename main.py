import os

from happytransformer import HappyTextToText, TTEvalArgs, TTTrainArgs

from common import (
    clear_all_csv_files,
    load_datasets,
    logger,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    MODEL_SAVE_DIR,
)

T5_MODEL = (
    "vennify/t5-base-grammar-correction"  # t5-small, t5-base, t5-large, t5-3b, t5-11b
)

TRAIN_DATASETS_FILE_PATH = os.path.join(os.getcwd(), "data", "train")
TRAIN_DATASETS_FILE_NAME_PREFIXES = [
    "fce.train.gold.bea19",
    "fce.dev.gold.bea19",
    # "lang8.train.auto.bea19",
    "nucle.train.gold.bea19",
    "A.train.gold.bea19",
    "B.train.gold.bea19",
    "C.train.gold.bea19",
]

DEV_DATASETS_FILE_PATH = os.path.join(os.getcwd(), "data", "dev")
DEV_DATASETS_FILE_NAME_PREFIXES = ["ABCN.dev.gold.bea19"]


def eval_model(happy_tt, csv_paths, print_prefix=""):
    """
    Evaluate model on given csv files
    """
    args = TTEvalArgs(
        max_input_length=MAX_INPUT_LENGTH,
        max_output_length=MAX_OUTPUT_LENGTH,
    )

    for csv_path in csv_paths:
        result = happy_tt.eval(
            csv_path,
            args=args,
        )
        logger.info(f"{print_prefix} {result}")


def train_model(happy_tt, csv_path, index=0):
    """
    Train model on given csv files
    """
    args = TTTrainArgs(
        batch_size=8,
        max_input_length=MAX_INPUT_LENGTH,
        max_output_length=MAX_OUTPUT_LENGTH,
        num_train_epochs=1,
    )

    logger.info(f"{index + 1}. Training on {csv_path}")
    happy_tt.train(csv_path, args=args)
    logger.info("--------------------------------------------------")


# ---------------------------------------------------------------------------- #

# Clear all csv files in the data folder
clear_all_csv_files([TRAIN_DATASETS_FILE_PATH, DEV_DATASETS_FILE_PATH])

# Load model
logger.info(f"Loading T5 model: {T5_MODEL}")
model = HappyTextToText("T5", T5_MODEL)

# ---------------------------------------------------------------------------- #

# Load dev and train datasets
logger.info(f"Loading dev datasets...")
dev_csv_paths = load_datasets(
    DEV_DATASETS_FILE_PATH,
    DEV_DATASETS_FILE_NAME_PREFIXES,
    has_labels=[True],
    detokenize_input=True,
)

logger.info(f"Loading train datasets...")
train_csv_paths = load_datasets(
    TRAIN_DATASETS_FILE_PATH,
    TRAIN_DATASETS_FILE_NAME_PREFIXES,
    has_labels=[True] * len(TRAIN_DATASETS_FILE_NAME_PREFIXES),
    detokenize_input=True,
)

# ---------------------------------------------------------------------------- #

# Evaluate model before training
logger.info(f"Evaluating model on dev datasets...")
eval_model(model, dev_csv_paths, "Before training")

# ---------------------------------------------------------------------------- #

# Train model
logger.info(f"Start training...")
for i, train_csv_path in enumerate(train_csv_paths):
    train_model(model, train_csv_path, i)
logger.info(f"Training completed")

# ---------------------------------------------------------------------------- #

# Evaluate model after training
logger.info(f"Evaluating model on dev datasets...")
eval_model(model, dev_csv_paths, "After training")

# ---------------------------------------------------------------------------- #

# Save model
model.save(MODEL_SAVE_DIR)
