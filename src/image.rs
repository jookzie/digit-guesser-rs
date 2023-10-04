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
        let line = vec!["─"; self.columns * 2].into_iter().collect::<String>(); 

        println!("╭{}╮", &line);
        
        for (i, pixel) in self.pixels.iter().enumerate() {
            if i == 0 {
                print!("│")
            } else if i % self.columns == 0 {
                print!("│\n│")
            }
            if *pixel == 0 {
                print!("⬜")  
            } else {
                print!("⬛")
            }
            if i == self.pixels.len() - 1 {
                print!("│")
            }
        }

        println!("\n╰{}╯", &line);
    }
}
