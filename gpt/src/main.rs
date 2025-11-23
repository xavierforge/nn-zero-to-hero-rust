use std::fs;

use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::{Int, Tensor};

use crate::dataset::{Dataset, Split};
use crate::tokenizer::Tokenizer;

mod dataset;
mod tokenizer;

type MyBackend = Wgpu;

fn main() {
    let text = fs::read_to_string("./input.txt").expect("Should be able to read the file");
    println!("=== Dataset Statistics ===");
    println!("Total characters: {}", text.len());
    println!("Preview (first 100 chars): {}\n", &text[..100]);

    let shakespeare_tokenizer = Tokenizer::new(&text);
    let chars = shakespeare_tokenizer.get_vocab();
    let vocab_size = shakespeare_tokenizer.vocab_size();
    println!("=== Tokenizer Information ===");
    println!("Vocabulary: {}", chars.iter().collect::<String>());
    println!("Vocabulary size: {}\n", vocab_size);

    println!("=== Tokenizer Test ===");
    let test_text = "hii there";
    let encoded = shakespeare_tokenizer
        .try_encode(test_text)
        .expect("Test text contains characters not in training vocabulary");
    println!("Input text: \"{}\"", test_text);
    println!("Encoded: {:?}", encoded);
    println!("Decoded: \"{}\"\n", shakespeare_tokenizer.decode(&encoded));

    let device = Default::default();
    MyBackend::seed(&device, 1337);
    let tokens = shakespeare_tokenizer.encode(&text);
    let data = Tensor::<MyBackend, 1, Int>::from_data(tokens.as_slice(), &device);
    println!("=== Tensor Information ===");
    println!("Shape: {:?}", data.shape());
    println!("Data type: {:?}", data.dtype());
    println!("First 10 tokens: {}", data.clone().slice(..10));

    let dataset = Dataset::new(data);
    println!("=== Dataset Split Information ===");
    println!("Training set size:   {} tokens", dataset.train_size());
    println!("Validation set size: {} tokens\n", dataset.val_size());

    let block_size = 8;
    let batch_size = 4;
    let (inputs, targets) = dataset.get_batch(block_size, batch_size, Split::Training, &device);
    println!("=== Batch Example ===");
    println!("Block size: {}", block_size);
    println!("Batch size: {}\n", batch_size);
    println!("Input sequences shape: {:?}", inputs.shape());
    println!("{}", inputs);
    println!("Target sequences shape: {:?}", targets.shape());
    println!("{}", targets);
}
