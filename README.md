Gamma 270M & 270B – Pretraining Pipeline

This project documents the Gamma 270M / 270B model pretraining process, inspired by Vizura’s tutorials and implemented step by step.

📚 Dataset

Tiny Stories dataset (~2M rows, 20K stories)

Tokenized and stored in train.bin and val.bin using np.memmap for fast loading.

🧠 Tokenization

Character-level: simple but inefficient (ballooning effect, loses semantics).

Word-level: limited vocabulary, fails on typos.

Subword Tokenization (BPE) ✅: merges frequent characters, efficient and robust.

🔄 Input–Output Pairs

Converts token sequences into next-token prediction targets.

Example: [1, 11, 15, 24] → [11, 15, 24, 63]

Uses pinned memory + non-blocking transfer for training efficiency.

🏗 Model Architecture

Gamma uses 18 Transformer blocks (3 full attention + 15 sliding attention).

Key Components

Token Embeddings → convert token IDs to trainable vectors

RMSNorm → normalizes activations

Multi-Query & Sliding Attention → efficient context handling

Rotary Positional Encoding (RoPE) → adds position information

Feed Forward Network → expands/activates dimensions (640 → 2048 → 640)

Output Projection → maps back to vocabulary space

⚙️ Training

Loss: Cross-entropy on next-token prediction

Optimizer: Adam + learning rate warmup/decay

Mixed precision training (FP16/FP32)

Gradient accumulation to simulate large batch sizes

🖥 Inference

Autoregressive text generation: model predicts next token, appends it, and continues until completion.
