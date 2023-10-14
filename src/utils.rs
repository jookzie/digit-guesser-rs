pub fn sigmoid(f: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-f))
}

pub fn sigmoid_derivative(f: f64) -> f64 {
    let value = sigmoid(f);
    value * (1.0 - value)
}