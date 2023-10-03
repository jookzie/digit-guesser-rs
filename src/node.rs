#[derive(Clone)]
pub struct Node {
    value: f64,
    bias: f64,
}

impl Node {
    pub fn empty() -> Self {
        Node { value: 0.0, bias: 0.0 }
    }
}