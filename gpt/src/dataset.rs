use burn::prelude::*;

const TRAIN_SPLIT_RATIO: f32 = 0.9;

pub struct Dataset<B: Backend> {
    train_data: Tensor<B, 1, Int>,
    val_data: Tensor<B, 1, Int>,
}

pub enum Split {
    Training,
    Validation,
}

impl<B: Backend> Dataset<B> {
    pub fn new(data: Tensor<B, 1, Int>) -> Self {
        let data_size = data.dims()[0];
        let training_size = (TRAIN_SPLIT_RATIO * data_size as f32) as usize;
        let train_data = data.clone().slice(..training_size);
        let val_data = data.clone().slice(training_size..);
        Self {
            train_data,
            val_data,
        }
    }

    pub fn train_size(&self) -> usize {
        self.train_data.dims()[0]
    }

    pub fn val_size(&self) -> usize {
        self.val_data.dims()[0]
    }

    pub fn get_batch(
        &self,
        block_size: usize,
        batch_size: usize,
        split: Split,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let (data, data_len) = match split {
            Split::Training => (self.train_data.clone(), self.train_size()),
            Split::Validation => (self.val_data.clone(), self.val_size()),
        };
        let max_start_index = (data_len - block_size) as f64;

        let random_indices: Tensor<B, 1, Int> = Tensor::random(
            [batch_size],
            burn::tensor::Distribution::Uniform(0.0, max_start_index),
            device,
        );

        let indices_data = random_indices.to_data();

        let input_sequences = indices_data
            .iter()
            .map(|start_idx: i32| {
                let end_idx = start_idx + block_size as i32;
                data.clone().slice(start_idx..end_idx)
            })
            .collect::<Vec<Tensor<B, 1, Int>>>();

        let target_sequences = indices_data
            .iter()
            .map(|start_idx: i32| {
                let shifted_start = start_idx + 1;
                let shifted_end = shifted_start + block_size as i32;
                data.clone().slice(shifted_start..shifted_end)
            })
            .collect::<Vec<Tensor<B, 1, Int>>>();

        let inputs = Tensor::stack(input_sequences, 0);
        let targets = Tensor::stack(target_sequences, 0);
        (inputs, targets)
    }
}
