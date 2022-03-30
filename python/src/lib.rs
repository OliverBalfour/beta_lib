
use beta_lib::{BetaDist, CosineInterpolatedDiscreteDist, Dist};
use pyo3::prelude::*;

#[pyfunction]
fn pdf(x: f64, a: f64, b: f64) -> PyResult<f64> {
    Ok(BetaDist::pdf(x, a, b))
}

#[pyfunction]
fn cdf(x: f64, a: f64, b: f64) -> PyResult<f64> {
    Ok(BetaDist::cdf(x, a, b))
}

#[pyfunction]
fn ppf(x: f64, a: f64, b: f64) -> PyResult<f64> {
    Ok(BetaDist::ppf(x, a, b))
}

#[pyfunction]
fn cosine_pdf(d: Vec<f64>, x: f64) -> PyResult<f64> {
    let dist = CosineInterpolatedDiscreteDist::from_sample(d);
    Ok(dist.pdf(x))
}

#[pyfunction]
fn cosine_cdf(d: Vec<f64>, x: f64) -> PyResult<f64> {
    let dist = CosineInterpolatedDiscreteDist::from_sample(d);
    Ok(dist.cdf(x))
}

#[pyfunction]
fn cosine_ppf(d: Vec<f64>, x: f64) -> PyResult<f64> {
    let dist = CosineInterpolatedDiscreteDist::from_sample(d);
    Ok(dist.ppf(x))
}

#[pymodule]
fn beta(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pdf, m)?)?;
    m.add_function(wrap_pyfunction!(cdf, m)?)?;
    m.add_function(wrap_pyfunction!(ppf, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_ppf, m)?)?;
    Ok(())
}

