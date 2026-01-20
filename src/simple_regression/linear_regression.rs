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
                println!("Epoch {}: Error MSE = {:.2} | w = {:.4}, b = {:.4}",
                    epoch, total_error / n, self.w, self.b)
            }


        }
    }
}


// Helper function
pub fn run() {
    // 1. Retrieve data
    let data = get_training_data();
    println!("Datos cargados: {} registros", data.x_train.len());

    // 2. Initialize model
    let mut model = LinearRegresion::new();

    // 3. Configure hyperparameters
    let epochs = 10000;
    let learning_rate = 0.00001;

    println!("Iniciando entrenamiento...");
    model.train(&data.x_train, &data.y_train, epochs, learning_rate);


    // 4. Probar con una casa nueva
    let test_area = 100.0;
    let predicted_price = model.predict(test_area);

    println!("--------------------------------");
    println!("RESULTADO FINAL:");
    println!("Modelo aprendido: Precio = ({:.2} * Area) + {:.2}", model.w, model.b);
    println!("Predicción para casa de 100m2: {:.2} Millones", predicted_price);
    println!("(Esperábamos ~450-500 millones según nuestra fórmula secreta)");
}