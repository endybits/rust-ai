pub fn flat_tensors() {
    // TENSOR PLANO EN HEAP (12 floats)
    // [0.1, 0.2, 0.3, 0.4,   0.5, 0.6, 0.7, 0.8,   0.9, 1.0, 1.1, 1.2]
    //  ˆ-- Embedding 1 --ˆ.  ˆ-- Embedding 2 --ˆ   ˆ-- Embedding 3 --ˆ
    let flat_tensor: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];

    // Diemensión de nuestros embeddings (d_model)
    let d_model = 4;
    
    // .chunk_exact crea un iterador de slices
    // No copia nada, te va dando ventanas de 4 en 4
    for (i, embedding_view) in flat_tensor.chunks_exact(d_model).enumerate() {
        // embedding_view es un slice &[f32] de  longitud 4.
        // Es una ventana temporar hacia el vector original
        println!("Token {}: {:#?}", i, embedding_view);
    }
}