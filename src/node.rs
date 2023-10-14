use std::vec;

use rand::{rngs::ThreadRng, Rng};

#[derive(Clone)]
pub struct Node {
    pub value: f64,
    pub value_raw: f64,
    pub bias: f64,
    pub weights: Vec<f64>
}

impl Node {
    pub fn new(weights_length: usize) -> Self {
        Self {
            value: 0.0,
            value_raw: 0.0,
            bias: 0.0,
            weights: vec![0.0; weights_length]
        }
    }
    pub fn random(weights_length: usize, rng: &mut ThreadRng) -> Self {
        Self {
            value: 0.0,
            value_raw: 0.0,
            bias: rng.gen(),
            weights: (0..weights_length).map(|_| rng.gen_range(-1.0..1.0)).collect()
        }
    }
}
