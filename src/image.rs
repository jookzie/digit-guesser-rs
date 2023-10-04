use std::fmt;

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
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut output = String::with_capacity((self.columns + 4) * (self.rows + 3) * 3);

        let line = vec!["─"; self.columns * 2].into_iter().collect::<String>(); 

        output.push_str(&format!("╭{}╮\n", &line));
        
        for (i, pixel) in self.pixels.iter().enumerate() {
            if i == 0 {
                output.push('│')
            } else if i % self.columns == 0 {
                output.push_str("│\n│")
            }
            if *pixel == 0 {
                output.push('⬜')  
            } else {
                output.push('⬛')  
            }
            if i == self.pixels.len() - 1 {
                output.push('│')
            }
        }

        output.push_str(&format!("\n╰{}╯\n", &line));

        write!(f, "{}", output)
    }
}

