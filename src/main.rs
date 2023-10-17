use std::ops::{Mul, Sub};
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

const CHUNKS: usize = 10;
const ITERATIONS: usize = 75;

fn main() {
    let train_images = load_dataset("datasets/train-images-idx3-ubyte");
    let train_labels = load_dataset("datasets/train-labels-idx1-ubyte");
    let test_images  = load_dataset("datasets/t10k-images-idx3-ubyte");
    let test_labels  = load_dataset("datasets/t10k-labels-idx1-ubyte");

    let mut train_images: Vec<Image> = parse_datasets(&train_images, &train_labels);
    let mut test_images: Vec<Image> = parse_datasets(&test_images, &test_labels);

    drop(train_labels);
    drop(test_labels);

    let rng = &mut thread_rng();

    let mut model = Model::new_random(rng);

    // Train the model and display progress of each step
    let mut highscore = 0.0;

    for i in 1..=ITERATIONS {
        let last_model = model.clone();

        train_images.shuffle(rng);
        // let time = Instant::now();
        train(&mut model, &train_images);
        // println!("Training time: {:.0?}", time.elapsed());

        println!("Iteration: {i}");
        
        //let avg_cost = rate_by_cost(&mut model, &test_images);
        //println!("    Average cost: {avg_cost:.5}");
        //println!("Average accuracy: {:.2}%", avg_cost.mul(10.0).sub(100.0).abs());

        let accuracy = rate_by_accuracy(&mut model, &test_images);
        println!(" Accuracy: {:.2}%", accuracy.mul(100.0));
        println!("Highscore: {:.2}%", highscore.mul(100.0));
        
        if accuracy > highscore {
            highscore = accuracy;
        } else {
            model = last_model;
        }
        println!()
    }

    test_images.shuffle(rng);
    visual_test(&mut model, &test_images);
}

// Back propagation
#[allow(non_snake_case)]
fn train(model: &mut Model, train_images: &[Image]) {
    for images_per_step in train_images.chunks(CHUNKS) {
        let mut changes = Model::default();

        for image in images_per_step {
            model.feed_forward(image);

            let mut desired_values = [0.0; 10];
            desired_values[image.label()] = 1.0;

            let cost_deriv: Vec<f64> = model.output_layer
                        .iter()
                        .zip(desired_values)
                        .map(|(n, v)| 2.0 * (n.value - v))
                        .collect();

            for ((aL, cL), dc) in model.output_layer.iter().zip(changes.output_layer.iter_mut()).zip(cost_deriv.iter()) {
                let z = sigmoid_derivative(aL.value_raw);

                cL.bias = z * dc / CHUNKS as f64;
                for (((aL_w, cL_w), aL_1), cL_1) in aL.weights.iter().zip(cL.weights.iter_mut()).zip(model.hidden_layer.iter()).zip(changes.hidden_layer.iter_mut()) {
                    *cL_w = z * dc * aL_1.value / CHUNKS as f64; 
                    cL_1.value += z * dc * *aL_w / CHUNKS as f64;
                }
            }

            for ((aL, cL), d) in model.hidden_layer.iter().zip(changes.hidden_layer.iter_mut()).zip(cost_deriv) {
                let z = sigmoid_derivative(aL.value_raw);

                cL.bias = z * d / CHUNKS as f64;
                for (aL_1, cL_w) in model.input_layer.iter().zip(cL.weights.iter_mut()) {
                    *cL_w = z * d * aL_1.value / CHUNKS as f64;
                }
            }
        }
        *model -= changes;
    }
}


fn visual_test(model: &mut Model, test_images: &Vec<Image>) {
    for image in test_images {
        let guess = model.feed_forward(image);
        
        let mut desired_values = [0.0; 10];
        desired_values[image.label()] = 1.0;
        let cost = model.output_layer.cost(&desired_values);

        println!("{image}");
        println!("{}", model.output_layer);
        println!("   Guess: {guess}");
        println!("  Number: {}", image.label());
        println!("    Cost: {cost:.5}");
        println!("Accuracy: {:.2}%", cost.mul(10.0).sub(100.0).abs());

        thread::sleep(Duration::from_secs(4));
    }
}

// Returns the average cost of the model through all test samples
fn rate_by_cost(model: &mut Model, test_images: &[Image]) -> f64 {
    let mut average_cost = 0.0;

    for image in test_images.iter() {
        model.feed_forward(image);

        let mut desired_values = [0.0; 10];
        desired_values[image.label()] = 1.0;
        average_cost += model.output_layer.cost(&desired_values);
    }

    average_cost / (test_images.len() as f64)
}

/// Returns a rating between 0 and 1, depending on total correct guesses
fn rate_by_accuracy(model: &mut Model, test_images: &[Image]) -> f64 {
    let mut correct_guesses = 0.0;

    for image in test_images.iter() {
        if image.label() == model.feed_forward(image) {
            correct_guesses += 1.0;
        }
    }

    correct_guesses / (test_images.len() as f64)
}