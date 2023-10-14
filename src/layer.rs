use core::fmt;
use std::{ops::{Add, Deref, DerefMut, SubAssign, AddAssign}, fmt::Formatter};

use rand::rngs::ThreadRng;

use crate::{node::Node, utils::sigmoid};

#[derive(Clone)]
pub struct Layer {
    nodes: Vec<Node>,
    node_length: usize
}

impl Layer {
    pub fn new(length: usize, weights_per_node: usize) -> Self {
        Self {
            nodes: vec![Node::new(weights_per_node); length],
            node_length: weights_per_node
        }
    }

    pub fn random(length: usize, weights_per_node: usize, rng: &mut ThreadRng) -> Self {
        Self {
            nodes: (0..length)
                   .map(|_| Node::random(weights_per_node, rng))
                   .collect(),
            node_length: weights_per_node
        }
    }

    pub fn evaluate(&mut self, other: &Layer)  {
        assert_eq!(self.node_length, other.len());

        self.iter_mut()
            .for_each(|node| {
                node.value_raw = 
                    other.iter()
                         .zip(node.weights.iter())
                         .map(|(other_node, w)| other_node.value * *w)
                         .sum::<f64>()
                         .add(node.bias);
                node.value = sigmoid(node.value_raw);
            })
    }

    pub fn cost(&self, desired_values: &[f64]) -> f64 {
        self.iter()
            .zip(desired_values.iter())
            .map(|(node, desired_value)| (node.value - desired_value) * (node.value - desired_value))
            .sum::<f64>()
            
    }

    pub fn cost_derivative(&self, desired_values: &[f64]) -> f64 {
        self.iter()
            .zip(desired_values.iter())
            .map(|(node, desired_value)| 2.0 * (node.value - desired_value))
            .sum::<f64>()
    }
}

impl Deref for Layer {
    type Target = Vec<Node>;

    fn deref(&self) -> &Self::Target {
        &self.nodes
    }

}
impl DerefMut for Layer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.nodes
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}",
            self.iter()
                .enumerate()
                .map(|(i, node)| format!("{:2}: {:.2}\n", i, node.value).to_string())
                .collect::<String>()
        )
    }
}

impl SubAssign for Layer {
    fn sub_assign(&mut self, other: Self) {
        for (n1, n2) in self.iter_mut().zip(other.iter()) {
            n1.value -= n2.value;
            n1.value_raw -= n2.value_raw;
            n1.bias -= n2.bias;
            n1.weights.iter_mut()
                      .zip(n2.weights.iter())
                      .for_each(|(w1, w2)| *w1 -= *w2);
        }
    }
}
impl AddAssign for Layer {
    fn add_assign(&mut self, other: Self) {
        for (n1, n2) in self.iter_mut().zip(other.iter()) {
            n1.value += n2.value;
            n1.value_raw += n2.value_raw;
            n1.bias += n2.bias;
            n1.weights.iter_mut()
                      .zip(n2.weights.iter())
                      .for_each(|(w1, w2)| *w1 += *w2);
        }
    }
}