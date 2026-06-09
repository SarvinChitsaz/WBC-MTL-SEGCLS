import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

CSV1_PATH = os.path.join(DATA_DIR, "Class Labels of Dataset 1.csv")
CSV2_PATH = os.path.join(DATA_DIR, "Class Labels of Dataset 2.csv")

DATASET1_DIR = os.path.join(DATA_DIR, "Dataset 1")
DATASET2_DIR = os.path.join(DATA_DIR, "Dataset 2")

EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 8
IMAGE_SIZE = 256
CLS_LOSS_WEIGHT = 0.7
TEST_SIZE = 0.2
