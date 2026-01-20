use rand::Rng;
pub struct Dataset {
    pub x_train: Vec<f64>,
    pub y_train: Vec<f64>,
}

// 1. Estuctura acoplada
// Simulamos la llegada de datos como de una database o archivo CSV

struct RawRecord {
    area: f64,
    price: f64,
}

fn generate_mock_records(count: usize) -> Vec<RawRecord> {
    let mut records = Vec::new();
    let mut rng = rand::rng();

    for _ in 0..count {
        // Simulamos area entre 34 y 250 m2
        let area: f64 = rng.random_range(34.0..250.0);
        // Precio igual a area por 4.5 millones + ruido aleatorio + bias
        let noise: f64 = rand::random::<f64>() - 0.5;
        let price = area * 4.5 + noise + 20.0;
        records.push(RawRecord { area, price });
    }
    records
}


pub fn get_training_data() -> Dataset {
    // Simulamos la carga de datos
    let raw_data = generate_mock_records(100000000);

    // ETL: Extraer las características y etiquetas
    let x_train: Vec<f64> = raw_data.iter().map(|record| record.area).collect();
    let y_train: Vec<f64> = raw_data.iter().map(|record| record.price).collect();

    Dataset { x_train, y_train }
}