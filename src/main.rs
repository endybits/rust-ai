mod simple_regression;
mod slices_practice;

// use crate::simple_regression::linear_regression::run as execute_linear_regression;

// fn main() {
//     execute_linear_regression();
// }

use crate::slices_practice::context_window_1::context_window;
use crate::slices_practice::flat_tensors_2::flat_tensors;

fn main() {
    context_window();
    flat_tensors();
}