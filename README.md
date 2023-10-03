# Digit guesser 
The project is a rust implementation of guessing hand-written digits using a machine learning algorithm, with the help of the [MNIST database](http://yann.lecun.com/exdb/mnist/).
It has been heavily inspired by [3Blue1Brown's video](https://www.youtube.com/watch?v=aircAruvnKk) introduction to deep learning.

This project's main purpose personally is to learn more about rust, hence the minimal amount of crates.

# Running
1. Install [rust](https://www.rust-lang.org/)
2. Clone this repository
```
git clone https://github.com/jookzie/digit-guesser-rs.git --depth=1
cd digit-guesser-rs
```
3. Run the project
```
cargo run --release
```