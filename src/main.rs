mod simple_regression;
mod slices_practice;

// use crate::simple_regression::linear_regression::run as execute_linear_regression;

// fn main() {
//     execute_linear_regression();
// }

use crate::slices_practice::context_window_1::context_window;
use crate::slices_practice::flat_tensors_2::flat_tensors;
use crate::slices_practice::process_attention_03::{generate_weights, process_attention_layer};
use crate::slices_practice::mutable_slice_in_place_4::apply_relu_in_place;

fn main() {
    context_window();
    flat_tensors();

    let all_weights = generate_weights(3000);
    process_attention_layer(&all_weights);

    let mut all_weights_for_relu = generate_weights(20);
    println!("\n\nPesos antes de ReLU: {:#?}", all_weights_for_relu);

    // Aplicar ReLU in-place
    apply_relu_in_place(&mut all_weights_for_relu);
    println!("\n\nPesos después de ReLU: {:#?}", all_weights_for_relu);
}