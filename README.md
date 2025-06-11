# Neural Networks: Zero to Hero (Rust Edition)

Inspired by [Andrej Karpathy's Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero), this is a Rust implementation of the same learning journey.

The goal is to implement small neural network projects in Rust, using modern Rust ML libraries like:

- [Candle](https://github.com/huggingface/candle): minimal tensor library for Rust, no dependencies on PyTorch / C++
- [Burn](https://github.com/tracel-ai/burn): flexible, modular deep learning framework in Rust

---

## Project Structure

This repository is a **Cargo workspace** with multiple independent projects:

| Project   | Description                     | Run command |
|-----------|---------------------------------|-------------|
| `micrograd` | Scalar-based autograd engine (warmup) | `cargo run -p micrograd` |
| `makemore`  | Character-level language model / MLP | `cargo run -p makemore`  |
| `gpt`       | Basic GPT implementation        | `cargo run -p gpt`       |
| `gpt2`      | Improved GPT-2 implementation   | `cargo run -p gpt2`      |
| `common`    | Shared utilities (tensor wrappers, tokenizer, data loaders) | used as library crate |

---

## Philosophy

- Each project is **independent** — you can run them individually.
- `common/` contains **shared code**, such as tokenizer utilities, tensor abstractions, and data loaders.
- Tensor operations will primarily use **Candle** or **Burn**, not re-implemented manually.
- Projects aim to be idiomatic Rust and fast.

---

## Running Projects

To run a project:
```bash
cargo run -p <project_name>
```

Example:
```bash
cargo run -p micrograd
```
---
Roadmap

✅ Setup workspace and project structure

⬜ Implement basic micrograd with manual backward pass

⬜ Implement makemore char-level model

⬜ Implement GPT with attention

⬜ Implement GPT-2 with optimizations

⬜ Experiment with Burn and Candle interop

⬜ Benchmark performance

---

Notes

    This is a learning project — expect rapid iteration and refactoring.

    Feedback and contributions welcome!
