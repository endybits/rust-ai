mod simple_regression;

use crate::simple_regression::linear_regression::train;
use crate::simple_regression::data::get_training_data;

fn main() {
    println!("Hello, world!");
    train();
    let dataset = get_training_data();

    println!("Training data:");
    for (x, y) in dataset.x_train.iter().zip(dataset.y_train.iter()) {
        println!("Area: {:.2} m2, Price: {:.2} millions", x, y);
    }
}
