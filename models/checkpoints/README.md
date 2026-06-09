# 💾 Model Checkpoint (Pretrained Weights)

This project does NOT include the trained model checkpoint due to file size limitations.

---

# 📥 Step 1 — Download Model

Download the pretrained checkpoint from Google Drive:

https://drive.google.com/file/d/1BCZJtTNnL3xxWzJjYKjZZIO9T2th2zUd/view?usp=sharing

After downloading, you will get this file:

wbc_multitask_checkpoint.ckpt

---

# 📁 Step 2 — Place File in Correct Directory

Move the downloaded file into your project like this:

```bash

WBC-Multitask-CLS/
└── models/
    └── checkpoints/
        └── wbc_multitask_checkpoint.ckpt
```

If the folder does not exist, create it:

mkdir -p models/checkpoints

---

# 🚀 Step 3 — Load Checkpoint in Python

Use the following code:

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(
    "models/checkpoints/wbc_multitask_checkpoint.ckpt",
    map_location=device
)

print("Checkpoint loaded successfully")

---

# 🧠 Step 4 — Restore Model Weights

After loading checkpoint, restore model:

model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

---

# 📦 What is inside the checkpoint?

The file contains:

- Trained model weights
- Optimizer state
- Training epoch
- Evaluation metrics (Accuracy, F1, Dice, etc.)
- Class label mappings
- Model configuration

---

# ⚠️ Important Notes

- Make sure file path is exactly correct
- Use same model architecture when loading weights
- GPU is recommended for inference
