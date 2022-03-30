
use beta_lib as beta;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn pdf(x: f64, a: f64, b: f64) -> f64 {
    beta::pdf(x, a, b)
}

#[wasm_bindgen]
pub fn cdf(x: f64, a: f64, b: f64) -> f64 {
    beta::cdf(x, a, b)
}

#[wasm_bindgen]
pub fn ppf(x: f64, a: f64, b: f64) -> f64 {
    beta::ppf(x, a, b)
}
