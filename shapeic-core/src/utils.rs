pub fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    match count {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (count - 1) as f64;
            (0..count)
                .map(|index| {
                    if index + 1 == count {
                        stop
                    } else {
                        start + step * index as f64
                    }
                })
                .collect()
        }
    }
}
