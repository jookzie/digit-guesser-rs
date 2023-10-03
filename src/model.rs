use crate::node::Node;
use crate::image::Image;

#[derive(Clone)]
struct Model {
    weights:        Vec<f64>,
    biases:         Vec<f64>,
    height:         usize,
    width:          usize,
    input_height:   usize,
    output_height:  usize
}

impl Model {
    pub fn new(
        input_height: usize,
        width: usize, 
        height: usize, 
        output_height: usize
    ) -> Self {
        let total_weights: usize = usize::pow(height, width as u32) + height * (input_height + output_height);
        let weights: Vec<f64> = vec![0.0; total_weights];

        let total_biases: usize = height * width + output_height;
        let biases: Vec<f64> = vec![0.0; total_biases];

        Model {
            weights,
            biases,
            width,
            height,
            input_height,
            output_height
        }    
    }


    fn feed_forward(self, image: Image) -> Vec<Node> {
        let mut nodes = Vec::<Node>::with_capacity(self.total_nodes());

        todo!();
        nodes
    }
    
    fn back_propagate(&mut self, nodes: Vec<Node>) -> Self {
        todo!();
        self.clone()
    }

    fn total_nodes(&self) -> usize {
        self.height * self.width + self.input_height + self.output_height
    }


   // fn link_layers(layer1: Vec<Node>, layer2: Vec<Node>) -> Vec<Weight<'a>> {
   //     let mut weights = Vec::with_capacity(layer1.len() * layer2.len());

   //     for node1 in layer1 {
   //         for node2 in layer2 {
   //             weights.push(Weight::new(0.0, &node1, &node2)) 
   //         }
   //     }

   //     weights
   // }
}