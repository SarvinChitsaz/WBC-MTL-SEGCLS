import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")

dataset1_dir = os.path.join(DATA_DIR, "Dataset 1")
dataset2_dir = os.path.join(DATA_DIR, "Dataset 2")

csv1_path = os.path.join(DATA_DIR, "Class Labels of Dataset 1.csv")
csv2_path = os.path.join(DATA_DIR, "Class Labels of Dataset 2.csv")
