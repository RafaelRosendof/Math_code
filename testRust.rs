
//use rayon::prelude::*;

fn trapezoidal<F>(a: f64 , b: f64 , f: F , n: usize) -> f64
where F: Fn(f64) -> f64 + Sync,

{
    let h = (b - a) / n as f64;

    let mut integral = (f(a) + f(b)) / 2.0;

    for i in 1..n {
        integral += f(a + i as f64 * h);
    }
    integral *= h;

    integral
}

fn trapezoidal_erro<F>(f: F , a: f64 , b: f64 , mut n: usize , tol: f64) -> f64
where F: Fn(f64) -> f64 + Sync,
{
    let mut integral_prev = trapezoidal(a, b, &f, n);

    loop{
        n *= 2;
        let integral = trapezoidal(a, b, &f, n);

        if (integral - integral_prev).abs() < tol {
            return integral;
        }

        integral_prev = integral; //atualiza 
    }
}


fn simpson38<F>(a: f64 , b: f64 , f: F , n: usize) -> f64
where
    F: Fn(f64) -> f64 + Sync,

{
    assert!(n % 3 == 0, "n must be a multiple of 3");

    let  h = (b - a) / n as f64;

    let mut integral = f(a) + f(b);

    for i in 1..n{

        if i % 3 == 0 {
            integral += 2.0 * f(a + i as f64 * h);
        }
        else {
         integral += 3.0 * f(a + i as f64 * h);
        }
    }

    integral = 3.0 * h * integral/8.0;

    integral

}

fn main(){

    //let f = |x: f64| -> f64 { (x * x + x).sqrt().(+ 1.0).log10()};
    let f = |x: f64| -> f64 { ((x * x + x).sqrt() + 1.0).log10() };



    let a = 2.0;
    let b = 100.0;
    let n = 60;
    let tol = 1e-10;

    let res_trap = trapezoidal_erro(&f, a, b, n, tol);

    println!("Trapezoidal: {}", res_trap);

    let res_simp = simpson38(a, b, &f, n);
    println!("Simpson 3/8: {}", res_simp);

    
}