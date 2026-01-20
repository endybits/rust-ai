mod simple_regression;

use crate::simple_regression::linear_regression::run as execute_linear_regression;
// use crate::simple_regression::data::get_training_data;

fn main() {
    execute_linear_regression();
}
