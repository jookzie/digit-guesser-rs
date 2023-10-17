use std::ops::{SubAssign, AddAssign};

use rand::rngs::ThreadRng;

use crate::{layer::Layer, image::Image};

#[derive(Clone)]
pub struct Model {
    pub input_layer: Layer,
    pub hidden_layer: Layer,
    pub output_layer: Layer
}

impl Model {
    pub fn new_random(rng: &mut ThreadRng) -> Self {
        Self {
            input_layer: Layer::random(784, 0, rng),
            hidden_layer: Layer::random(16, 784, rng),
            output_layer: Layer::random(10, 16, rng),
        }
    }

    pub fn feed_forward(&mut self, image: &Image) -> usize {
        for (node, pixel) in self.input_layer.iter_mut().zip(image.iter()) {
            node.value = f64::from(*pixel);
        }

        self.hidden_layer.evaluate(&self.input_layer);
        self.output_layer.evaluate(&self.hidden_layer);

        self.output_layer
            .iter()
            .enumerate()
            .max_by(|(_, n1), (_, n2)| n1.value.total_cmp(&n2.value))
            .unwrap()
            .0
    }
}

impl Default for Model {
    fn default() -> Self {
        Self {
            input_layer: Layer::new(784, 0),
            hidden_layer: Layer::new(16, 784),
            output_layer: Layer::new(10, 16)
        }
    }
}

impl SubAssign for Model {
    fn sub_assign(&mut self, other: Self) {
        self.input_layer -= other.input_layer;
        self.hidden_layer -= other.hidden_layer;
        self.output_layer -= other.output_layer;
    }
}
impl AddAssign for Model {
    fn add_assign(&mut self, other: Self) {
        self.input_layer += other.input_layer;
        self.hidden_layer += other.hidden_layer;
        self.output_layer += other.output_layer;
    }
}