pub fn apply_relu_in_place(tensor_data: &mut [f32]) {
    for val in tensor_data.iter_mut() {
        if *val < 0.0 {
            *val = 0.0;
        }
    }
}