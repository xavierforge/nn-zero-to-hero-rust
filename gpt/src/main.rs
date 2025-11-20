use std::fs;

use burn::backend::Wgpu;
use burn::tensor::{Int, Tensor};

use crate::tokenizer::Tokenizer;

mod tokenizer;

type Backend = Wgpu;

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
    let tokens = shakespeare_tokenizer.encode(&text);
    let data = Tensor::<Backend, 1, Int>::from_data(tokens.as_slice(), &device);
    println!("=== Tensor Information ===");
    println!("Shape: {:?}", data.shape());
    println!("Data type: {:?}", data.dtype());
    println!("First 10 tokens: {}", data.slice(..10));
}
