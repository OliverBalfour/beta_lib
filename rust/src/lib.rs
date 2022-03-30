use statrs::distribution::{Beta,Continuous,ContinuousCDF};

pub fn pdf(x: f64, a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || a > 10000.0 || b > 10000.0 {
        println!("Invalid beta.pdf values: alpha={}, beta={}, x={}", a, b, x)
    }
    let p = Beta::new(a, b).unwrap();
    p.pdf(x)
}

pub fn cdf(x: f64, a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || a > 10000.0 || b > 10000.0 {
        println!("Invalid beta.cdf values: alpha={}, beta={}, x={}", a, b, x)
    }
    let p = Beta::new(a, b).unwrap();
    p.cdf(x)
}

pub fn ppf(x: f64, a: f64, b: f64) -> f64 {
    // inverse CDF / percentile point function / quantile function
    if a <= 0.0 || b <= 0.0 || a > 10000.0 || b > 10000.0 {
        println!("Invalid beta.ppf values: alpha={}, beta={}, x={}", a, b, x)
    }
    let p = Beta::new(a, b).unwrap();
    p.inverse_cdf(x)
}

