use core::f64;

use crate::simple_regression::data::get_training_data;

// This struct is our neuron
struct LinearRegresion {
    w: f64,
    b: f64,
}

impl LinearRegresion {
    // Constructor. It borns very fulished
    pub fn new() -> LinearRegresion {
        LinearRegresion { w: 0.0, b: 0.0 }
    }

    // Inference the formula y = wx + b
    pub fn predict(&self, x: f64) -> f64 {
        (self.w * x) + self.b
    }

    // The learning gym
    // &mut we need permission to modify w and b
    pub fn train(&mut self, x_data: &Vec<f64>, y_data: &Vec<f64>, epochs: usize, learning_rate: f64) {
        let n = x_data.len() as f64;

        for epoch in 0..epochs {
            let mut sum_grad_w = 0.0;
            let mut sum_grad_b = 0.0;
            let mut total_error = 0.0;

            // Iterate the data for each house
            for i in 0..x_data.len() {
                let x = x_data[i]; // Area
                let y = y_data[i]; // Price

                // 1. Forward pass (predict)
                let prediction = self.predict(x);

                // 2. Calculate the error (predition - reality)
                let error = prediction - y;

                // Cummulate gradient
                // How much did x contribute to the error?
                sum_grad_w += error * x;
                sum_grad_b += error;

                // Mean Squared Error MSE
                total_error += error.powi(2);
            }

            // Backward Pass (update weights)
            // Promediar los gradientes y moverse en contra del error
            // Multiplicamos por 2/N (derivada del MSE)
            let dw = (2.0 / n) * sum_grad_w;
            let db = (2.0 / n) * sum_grad_b;

            self.w -= learning_rate * dw;
            self.b -= learning_rate * db;

            // Print the progres each 100 epoch
            if epoch % 100 == 0 {
                println!("Epoch {}: Error MSE = {:.4} | w = {:.4}, b = {:.4}",
                    epoch, total_error / n, self.w, self.b)
            }


        }
    }
}


// Helper function
pub fn run() {
    // 1. Cargar datos
    let data = get_training_data();
    
    // --- PASO A: Normalizar X (Área) ---
    let x_min = data.x_train.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = data.x_train.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    let x_norm: Vec<f64> = data.x_train.iter()
        .map(|x| (x - x_min) / (x_max - x_min))
        .collect();

    // --- PASO B: Normalizar Y (Precio) ---
    // Esto es clave para que el modelo converja rápido
    let y_min = data.y_train.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = data.y_train.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let y_norm: Vec<f64> = data.y_train.iter()
        .map(|y| (y - y_min) / (y_max - y_min))
        .collect();
        
    println!("Escalas -> Área: {}-{} | Precio: {}-{}", x_min, x_max, y_min, y_max);

    // 2. Inicializar modelo
    let mut model = LinearRegresion::new();

    // 3. Entrenamiento
    // Como todo es 0-1, un LR de 0.1 es perfecto y rápido.
    let epochs = 2000; 
    let learning_rate = 0.1; 

    println!("Iniciando entrenamiento blindado...");
    // Entrenamos con AMBOS vectores normalizados
    model.train(&x_norm, &y_norm, epochs, learning_rate);

    // 4. Predicción Correcta (Pipeline de Inferencia)
    let test_area_real = 100.0;
    
    // a) Normalizamos el input (Llevamos 100 al mundo 0-1)
    let test_area_norm = (test_area_real - x_min) / (x_max - x_min);
    
    // b) El modelo predice en "idioma normalizado" (0-1)
    let predicted_norm = model.predict(test_area_norm);
    
    // c) Des-normalizamos el output (Llevamos el resultado al mundo real de Millones)
    let predicted_price = predicted_norm * (y_max - y_min) + y_min;

    println!("--------------------------------");
    println!("RESULTADO FINAL:");
    println!("Modelo interno (w, b): {:.4}, {:.4}", model.w, model.b);
    println!("Predicción para 100m2: {:.2} Millones", predicted_price);
}