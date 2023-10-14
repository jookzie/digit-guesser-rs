use std::ops::{Mul, Sub, Div};
use std::time::Instant;
use std::{time::Duration, thread};

mod parser;
mod image;
mod node;
mod layer;
mod utils;
mod model;

use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::image::Image;
use crate::model::Model;
use crate::parser::{load_dataset, parse_datasets};
use crate::utils::sigmoid_derivative;

const CHUNKS: usize = 1_000;

fn main() {
    let train_images = load_dataset("datasets/train-images-idx3-ubyte");
    let train_labels = load_dataset("datasets/train-labels-idx1-ubyte");
    let test_images  = load_dataset("datasets/t10k-images-idx3-ubyte");
    let test_labels  = load_dataset("datasets/t10k-labels-idx1-ubyte");

    let mut train_images: Vec<Image> = parse_datasets(train_images, train_labels);
    let mut test_images: Vec<Image> = parse_datasets(test_images, test_labels);

    let rng = &mut thread_rng();
    test_images.shuffle(rng);

    let mut model = Model::new_random(rng);

    let mut highscore = 10.0;

    for i in 0..100 {
        println!("Iteration {i}");

        let last_model = model.clone();

        train_images.shuffle(rng);
        train(&mut model, &train_images);
        
        let avg_cost = auto_rate(&mut model, &test_images);
        if avg_cost < highscore {
            highscore = avg_cost;
        } else {
            model = last_model;
            println!("Bad!")
        }
        println!()
    }

    //visual_test(&mut model, &test_images);
}

fn train(model: &mut Model, train_images: &Vec<Image>) {
    let time = Instant::now();

    for images in train_images[..60_000].chunks(CHUNKS) {
        let mut changes = Model::default();
        for image in images {
            model.feed_forward(image);
            let mut desired_values = [0.0; 10];
            desired_values[image.label()] = 1.0;

            let cost_deriv = model.output_layer.cost_derivative(&desired_values);

            for ((i, aL), cL) in model.output_layer.iter().enumerate().zip(changes.output_layer.iter_mut()) {
                let z = sigmoid_derivative(aL.value_raw);
                let delta_bias = z * cost_deriv;

                cL.bias += delta_bias / CHUNKS as f64;
                for (((aL_w, cL_w), aL_1), cL_1) in aL.weights.iter().zip(cL.weights.iter_mut()).zip(model.hidden_layer.iter()).zip(changes.hidden_layer.iter_mut()) {
                    *cL_w = delta_bias * aL_1.value / CHUNKS as f64; 
                    cL_1.value += delta_bias * *aL_w;
                }
            }
            for (aL, cL) in model.hidden_layer.iter().zip(changes.hidden_layer.iter_mut()) {
                let z = sigmoid_derivative(aL.value_raw);

                let delta_bias = z * cost_deriv / CHUNKS as f64;
                
                cL.bias += delta_bias;
                for (aL_1, cL_w) in model.input_layer.iter().zip(cL.weights.iter_mut()) {
                    *cL_w = delta_bias * aL_1.value;
                }
            }
        }
        *model -= changes;
    }
    println!("Training time: {:.0?}", time.elapsed());

}


fn visual_test(model: &mut Model, test_images: &Vec<Image>) {
    for image in test_images {
        let performance = model.feed_forward(&image);

        println!("{image}");
        //println!("{model}");
        println!("{}", model.output_layer);
        println!("Cost: {performance:.2}");
        println!("Performance: {:.1}%", performance.mul(10.0).powf(2.0).div(10.0).sub(100.0).abs());

        thread::sleep(Duration::from_secs(4));
    }

}
fn auto_rate(model: &mut Model, test_images: &Vec<Image>) -> f64 {
    let mut average_cost = 0.0;

    for image in test_images.iter() {
        average_cost += model.feed_forward(image);
    }

    average_cost /= test_images.len() as f64;

    println!("Average cost: {average_cost:.5}");
    println!("Average performance: {:.2}%", average_cost.mul(10.0).powf(2.0).div(10.0).sub(100.0).abs());

    average_cost
}
