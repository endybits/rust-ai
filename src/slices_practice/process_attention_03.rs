use rand::Rng;

pub fn generate_weights(count: usize) -> Vec<f32> {
    let mut weights: Vec<f32> = Vec::new();
    let mut range = rand::rng();
    for _ in 0..count {
        let weight: f32 = range.random_range(-1.0..1.0);
        weights.push(weight);
    }
    weights
}

pub fn process_attention_layer(all_weights: &[f32]) {
    // Supongamos que all_weights tiene 3000 floats.
    // Q: 1000 floats, K: 1000 floats, V: 1000 floats
    
    // Queremos separar Q, K, V usando slices
    // Primero tomamos Q y los separamos del resto
    // q_weights: puntero al inicio de all_weights, longitud 1000
    // rest: puntero al indice 1000 de all_weights, longitud 2000
    let (q_weights, rest) = all_weights.split_at(1000);

    // Ahora separamos K y V de rest
    // k_weights: puntero al inicio de rest, longitud 1000
    // v_weights: puntero al indice 1000 de rest, longitud 1000
    let (k_weights, v_weights) = rest.split_at(1000);

    // Resultado: (q_weights, k_weights, v_weights) son slices que apuntan a all_weights
    println!("Q weights slice length: {}", q_weights.len());
    println!("K weights slice length: {}", k_weights.len());
    println!("V weights slice length: {}", v_weights.len());


}