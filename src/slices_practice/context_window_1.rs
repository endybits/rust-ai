pub fn context_window() {
    // HEAP: se reservan los tokens en memoria. Digamos que en 0x1000
    let ctx_window: Vec<u32> = vec![101, 55, 12, 98, 27, 554, 34, 61, 109];

    // SLICE OPERATION 
    // Queremos los 4 ultimos tokens del dataset: [554, 34, 61, 109]
    let start_idx = ctx_window.len() - 4;
    let view: &[u32] = &ctx_window[start_idx..];

    // QUÉ PASÓ EN EL STACK
    // 'view' se crea instantáneamente:
    // Ptr: 0x1000 + (4 bytes * 4 offset) = 0x1010
    // Len: 3

    println!("Mirando tokens {:?}", view);
}