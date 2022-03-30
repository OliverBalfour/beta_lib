
use beta_lib as beta;
use beta::Dist;
use pyo3::prelude::*;

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
    fn new(samples: Vec<f64>) -> Self {
        let dist = beta::BetaDist::from_sample(&samples);
        Self { a: dist.a, b: dist.b, dist }
    }
    fn pdf(&self, x: f64) -> f64 { self.dist.pdf(x) }
    fn cdf(&self, x: f64) -> f64 { self.dist.cdf(x) }
    fn ppf(&self, p: f64) -> f64 { self.dist.ppf(p) }
    #[staticmethod] fn _pdf(x: f64, a: f64, b: f64) -> f64 { beta::BetaDist::from_parameters(a, b).pdf(x) }
    #[staticmethod] fn _cdf(x: f64, a: f64, b: f64) -> f64 { beta::BetaDist::from_parameters(a, b).cdf(x) }
    #[staticmethod] fn _ppf(p: f64, a: f64, b: f64) -> f64 { beta::BetaDist::from_parameters(a, b).ppf(p) }
}

#[pyclass]
struct CosineInterpolatedDiscrete {
    dist: beta::CosineInterpolatedDiscreteDist,
}

#[pymethods]
impl CosineInterpolatedDiscrete {
    #[new]
    fn new(samples: Vec<f64>) -> Self {
        Self { dist: beta::CosineInterpolatedDiscreteDist::from_sample(&samples) }
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

