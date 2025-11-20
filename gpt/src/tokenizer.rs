use std::collections::{BTreeSet, HashMap};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("Character '{0}' not in vocabulary")]
    CharacterNotInVocabulary(char),
    #[error("Index {0} out of bounds for vocabulary")]
    IndexOutOfBounds(usize),
}

pub struct Tokenizer {
    char_to_index: HashMap<char, usize>,
    index_to_char: Vec<char>,
}

impl Tokenizer {
    /// Creates a new tokenizer from the given text, building vocabulary from unique characters
    pub fn new(text: &str) -> Self {
        let chars: BTreeSet<char> = text.chars().collect();
        let index_to_char: Vec<char> = chars.iter().copied().collect();

        let char_to_index: HashMap<char, usize> = index_to_char
            .iter()
            .enumerate()
            .map(|(idx, c)| (*c, idx))
            .collect();

        Self {
            char_to_index,
            index_to_char,
        }
    }

    /// Returns the vocabulary as a sorted set of characters
    pub fn get_vocab(&self) -> BTreeSet<char> {
        self.index_to_char.iter().copied().collect()
    }

    /// Returns the size of the vocabulary
    pub fn vocab_size(&self) -> usize {
        self.index_to_char.len()
    }

    /// Encodes input text to indices, panics if any character is not in vocabulary
    pub fn encode(&self, input: &str) -> Vec<usize> {
        self.try_encode(input)
            .unwrap_or_else(|e| panic!("Encoding failed: {}", e))
    }

    /// Decodes indices to text, panics if any index is out of bounds
    pub fn decode(&self, input: &[usize]) -> String {
        self.try_decode(input)
            .unwrap_or_else(|e| panic!("Decoding failed: {}", e))
    }

    /// Encodes input text to indices, returning an error if any character is not in vocabulary
    pub fn try_encode(&self, input: &str) -> Result<Vec<usize>, TokenizerError> {
        input
            .chars()
            .map(|c| {
                self.char_to_index
                    .get(&c)
                    .copied()
                    .ok_or(TokenizerError::CharacterNotInVocabulary(c))
            })
            .collect()
    }

    /// Decodes indices to text, returning an error if any index is out of bounds
    pub fn try_decode(&self, input: &[usize]) -> Result<String, TokenizerError> {
        input
            .iter()
            .map(|&idx| {
                self.index_to_char
                    .get(idx)
                    .copied()
                    .ok_or(TokenizerError::IndexOutOfBounds(idx))
            })
            .collect()
    }
}
