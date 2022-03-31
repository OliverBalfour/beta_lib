
// Given nonlinear, continuous f : R^D->R and an initial R^D vector, find x in R^D that minimises f(x)
// Uses the Nelder-Mead method
pub fn nelder_mead_minimise<
    const D: usize,
    F: FnOnce([f64; D]) -> f64,
>(f: F, x0: [f64; D]) -> [f64; D] {
    let mut simplex: [[f64; D]; D] = ['hi'];
}
