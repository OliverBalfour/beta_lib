
use beta_lib as beta;
use beta::Dist;
use pyo3::prelude::*;
use pyo3::exceptions::*;

#[pyclass]
struct Beta {
    #[pyo3(get, set)]
    a: f64,
    #[pyo3(get, set)]
    b: f64,
    dist: beta::BetaDist,
}

#[pymethods]
impl Beta {
    #[new]
    fn new(samples: Vec<f64>) -> PyResult<Self> {
        match beta::BetaDist::from_sample(&samples) {
            Ok(dist) => Ok(Self { a: dist.a, b: dist.b, dist }),
            Err(s) => Err(PyValueError::new_err(s))
        }
    }
    fn pdf(&self, x: f64) -> f64 { self.dist.pdf(x) }
    fn cdf(&self, x: f64) -> f64 { self.dist.cdf(x) }
    fn ppf(&self, p: f64) -> f64 { self.dist.ppf(p) }
    #[staticmethod] fn _pdf(x: f64, a: f64, b: f64) -> f64 { beta::BetaDist::pdf(x, a, b) }
    #[staticmethod] fn _cdf(x: f64, a: f64, b: f64) -> f64 { beta::BetaDist::cdf(x, a, b) }
    #[staticmethod] fn _ppf(p: f64, a: f64, b: f64) -> f64 { beta::BetaDist::ppf(p, a, b) }
}

#[pyclass]
struct CosineInterpolatedDiscrete {
    dist: beta::CosineInterpolatedDiscreteDist,
}

#[pymethods]
impl CosineInterpolatedDiscrete {
    #[new]
    fn new(samples: Vec<f64>) -> PyResult<Self> {
        Ok(Self { dist: beta::CosineInterpolatedDiscreteDist::from_sample(&samples).unwrap() })
    }
    fn pdf(&self, x: f64) -> f64 { self.dist.pdf(x) }
    fn cdf(&self, x: f64) -> f64 { self.dist.cdf(x) }
    fn ppf(&self, p: f64) -> f64 { self.dist.ppf(p) }
}

#[pymodule]
fn beta(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Beta>()?;
    m.add_class::<CosineInterpolatedDiscrete>()?;
    Ok(())
}

