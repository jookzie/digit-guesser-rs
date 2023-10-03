use std::fs;

mod image;

use image::Image;

const MAGIC_IMAGE_NUMBER: u32 = 2051;
const MAGIC_LABEL_NUMBER: u32 = 2049;


fn parse_datasets(image_buffer: Vec<u8>, label_buffer: Vec<u8>) -> Vec<Image> {
    assert!(image_buffer.len() > 16, "Image buffer is less than header (16 bytes).");
    assert!(label_buffer.len() > 8, "Label buffer is less than header (8 bytes).");

    let magic_image_number: u32 = u32::from_be_bytes(image_buffer[0..4].try_into().unwrap());
    let image_count: u32        = u32::from_be_bytes(image_buffer[4..8].try_into().unwrap());
    let rows: usize             = u32::from_be_bytes(image_buffer[8..12].try_into().unwrap()) as usize;
    let columns: usize          = u32::from_be_bytes(image_buffer[12..16].try_into().unwrap()) as usize;

    let magic_label_number: u32 = u32::from_be_bytes(label_buffer[0..4].try_into().unwrap());
    let label_count: u32        = u32::from_be_bytes(label_buffer[4..8].try_into().unwrap());
    let labels: &[u8]           = &label_buffer[8..];

    assert!(magic_image_number == MAGIC_IMAGE_NUMBER, 
        "Expected magic image dataset number: {MAGIC_IMAGE_NUMBER}\nFound: {magic_image_number}");
    
    assert!(magic_label_number == MAGIC_LABEL_NUMBER, 
        "Expected magic image dataset number: {MAGIC_IMAGE_NUMBER}\nFound: {magic_label_number}");

    assert!(image_count == label_count, "Images do not match the number of labels: {image_count} - {label_count}");

    let mut images = Vec::<Image>::with_capacity(image_count as usize); 
    

    for i in 0..image_count as usize {

        let total_pixels: usize = rows * columns;

        let offset1: usize = 16usize + i * total_pixels;
        let offset2: usize = offset1 + total_pixels;

        images.push({
            let pixels = image_buffer[offset1..offset2].to_vec();
            let label = labels[i];
            Image::new(pixels, label, rows, columns)
        })
    };
    images
}


fn load_dataset(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|_| panic!("Could not find '{path}'."))
}

fn main() {
    let train_images = load_dataset("datasets/train-images-idx3-ubyte");
    let train_labels = load_dataset("datasets/train-labels-idx1-ubyte");
    let test_images  = load_dataset("datasets/t10k-images-idx3-ubyte");
    let test_labels  = load_dataset("datasets/t10k-labels-idx1-ubyte");

    let train_images: Vec<Image> = parse_datasets(train_images, train_labels);
    let test_images: Vec<Image> = parse_datasets(test_images, test_labels);





}
