
use beta_lib as beta;
use pyo3::prelude::*;

#[pyfunction]
fn pdf(x: f64, a: f64, b: f64) -> PyResult<f64> {
    Ok(beta::pdf(x, a, b))
}

#[pyfunction]
fn cdf(x: f64, a: f64, b: f64) -> PyResult<f64> {
    Ok(beta::cdf(x, a, b))
}

#[pyfunction]
fn ppf(x: f64, a: f64, b: f64) -> PyResult<f64> {
    Ok(beta::ppf(x, a, b))
}

#[pymodule]
fn beta(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pdf, m)?)?;
    m.add_function(wrap_pyfunction!(cdf, m)?)?;
    m.add_function(wrap_pyfunction!(ppf, m)?)?;
    Ok(())
}

