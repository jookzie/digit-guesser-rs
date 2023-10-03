pub struct Image {
    pixels: Vec<u8>,
    label: u8,
    rows: usize,
    columns: usize,
}

impl Image {  
    pub fn new(pixels: Vec<u8>, label: u8, rows: usize, columns: usize) -> Self {
        Image { pixels, label, rows, columns }
    }

    pub fn display(&self) {
        for i in 0..(self.rows * self.columns) {
            if self.pixels[i] == 0 {
                print!(" ")
            } else {
                print!("█")
            }
            if i % self.columns == 0 {
                println!()
            }
        }
    }
}
