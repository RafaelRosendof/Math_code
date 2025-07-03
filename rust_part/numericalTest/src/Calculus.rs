fn trapezoidal_integral(a: f64, b: f64, f: fn(f64) -> f64, tolerance: f64) -> f64 {
    let mut h = b - a;
    let mut n = 1;
    let mut prev = (f(a) + f(b)) * h / 2.0;

    loop {
        n *= 2;
        h /= 2.0;

        let mut sum = 0.0;

        for i in 1..n {
            if i % 2 != 0 {
                sum += f(a + i as f64 * h);
            }
        }

        let atual = prev / 2.0 + h * sum;
        if (atual - prev).abs() < tolerance {
            return atual;
        }
        prev = atual;
    }
}

fn simpson_integral(a: f64, b: f64, f: fn(f64) -> f64, tolerance: f64) -> f64 {
    let mut n = 2;
    let mut h = (b - a) / n as f64;
    let mut prev = (h / 3.0) * (f(a) + f(b) + 4.0 * f(a + h) + 2.0 * f(a + 2.0 * h));

    loop {
        n *= 2;
        h(b - a) / n as f64;

        let mut sum = f(a) - f(b);

        for i in 1..n {
            let mut x = a + i as f64 * h;

            if i % 2 == 0 {
                sum += 2.0 * f(x);
            } else {
                sum += 4.0 * f(x);
            }
        }

        let atual = (h / 3.0) * sum;
        if (atual - prev).abs() < tolerance {
            return atual;
        }
        prev = atual;
    }
}

fn simpson_integral_3_8(a: f64, b: f64, f: fn(f64), n: usize) -> f64 {
    if n % 3 != 0 {
        panic!("N precisa necessario ser multiplo de 3");
    }

    let h = (b - a) / n as f64;

    let mut sum = f(a) + f(b);

    for i in 1..n {
        let x = a + i as f64 * h;

        if i % 3 == 0 {
            sum += 2.0 * f(x);
        } else {
            sum += 3.0 * f(x);
        }
    }

    (3.0 * h / 8.0) * sum
}

// lim h -> 0 ( f(x) - f(x-h) / h) as follow the same 2 functions
fn numerical_differentiation(x: f64, h: f64, f: fn(f64) -> f64) -> f64 {
    (f(x) - f(x - h)) / h
}

fn forward_differentiation(a: f64, h: f64, f: fn(f64) -> f64, tolerance: f64) -> f64 {
    (f(x + h) - f(x - h)) / h
}

fn backward_differentiation(a: f64, h: f64, f: fn(f64) -> f64, tolerance: f64) -> f64 {
    (f(x + h) - f(x)) / h
}
