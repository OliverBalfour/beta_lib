use statrs::distribution::{Beta, Continuous, ContinuousCDF};
use std::vec::Vec;
mod statistics;
use statistics::*;
use std::f64::consts::PI;

pub trait Dist {
    // Fit a dist to a sample (eg with the method of moments)
    fn from_sample(d: Vec<f64>) -> Self;
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
    // inverse CDF / percentile point function / quantile function
    fn ppf(&self, p: f64) -> f64 {
        // Default implementation: we know cdf is monotonic so we can binary search
        let mut x = 0.5;
        let mut hi = 1.0;
        let mut lo = 0.0;
        loop {
            let d = self.cdf(x) - p;
            if d.abs() < 0.00001 { break x }
            if d < 0.0 {
                // cdf(x) < p so x is too low
                lo = x
            } else {
                hi = x
            }
            x = (hi + lo) / 2.0
        }
    }
}

pub struct BetaDist {
    imp: Beta,
    a: f64, // alpha
    b: f64, // beta
    m: Option<f64>, // mean of dist it was fit to (if applicable)
    v: Option<f64>, // variance
}

impl BetaDist {
    fn from_parameters(a: f64, b: f64, m: Option<f64>, v: Option<f64>) -> Self {
        if a <= 0.0 || b <= 0.0 || a > 1000.0 || b > 1000.0 {
            println!("Invalid Beta parameters: alpha={}, beta={}", a, b)
        }
        Self {
            imp: Beta::new(a, b).unwrap(),
            a: a.clamp(0.001, 1000.0),
            b: b.clamp(0.001, 1000.0),
            m, v,
        }
    }
    pub fn pdf(x: f64, a: f64, b: f64) -> f64 {
        Self::from_parameters(a, b, None, None).pdf(x)
    }
    pub fn cdf(x: f64, a: f64, b: f64) -> f64 {
        Self::from_parameters(a, b, None, None).cdf(x)
    }
    pub fn ppf(x: f64, a: f64, b: f64) -> f64 {
        Self::from_parameters(a, b, None, None).ppf(x)
    }
}

impl Dist for BetaDist {
    fn from_sample(d: Vec<f64>) -> Self {
        let (m, v) = d.mean_variance();
        let a = (m * (1.0 - m) / v - 1.0) * m;
        let b = (m * (1.0 - m) / v - 1.0) * (1.0 - m);
        Self::from_parameters(a, b, Some(m), Some(v))
    }
    fn pdf(&self, x: f64) -> f64 { self.imp.pdf(x) }
    fn cdf(&self, x: f64) -> f64 { self.imp.cdf(x) }
    fn ppf(&self, p: f64) -> f64 { self.imp.inverse_cdf(p) }
}

pub struct CosineInterpolatedDiscreteDist {
    // Take a discrete distribution of samples in the [0,1] interval
    //  and use cosine interpolation to create a continuous distribution
    //  with support on [0,1]
    // List of (x, p(x)) values sorted by x
    discrete: Vec<(f64, f64)>,
}

impl CosineInterpolatedDiscreteDist {
    pub fn cos_interp(x: f64, p1: (f64, f64), p2: (f64, f64)) -> f64 {
        // cosine interpolation between (x1,y1) and (x2,y2) at x1 <= x <= x2
        (p1.1 + p2.1) / 2.0 + (p1.1 - p2.1) / 2.0 * f64::cos(PI * (x - p1.0) / (p2.0 - p1.0))
    }
    pub fn cos_interp_int(x: f64, p1: (f64, f64), p2: (f64, f64)) -> f64 {
        // integral of cosine interpolation between (x1,y1) and (x2,y2) at x1 <= x <= x2
        (x - p1.0) * (p1.1 + p2.1) / 2.0 + (p1.1 - p2.1) * (p2.0 - p1.0) / 2.0 / PI
            * f64::sin(PI * (x - p1.0) / (p2.0 - p1.0))
    }
}

impl Dist for CosineInterpolatedDiscreteDist {
    fn from_sample(d: Vec<f64>) -> Self {
        let mut discrete: Vec<(f64, f64)> = Vec::with_capacity(10);
        // Count
        for p in d.iter() {
            for i in 0..discrete.len() {
                let (p2, c) = discrete[i];
                if *p == p2 {
                    // If it's in this spot, increment the counter
                    discrete[i] = (*p, c + 1.0);
                    break
                } else if *p < p2 {
                    // If we've passed where it should be, insert it there
                    discrete.insert(i, (*p, 1.0));
                    break
                }
            }
        }
        // Normalise
        let n = d.len() as f64;
        for i in 0..discrete.len() {
            let (p, c) = discrete[i];
            discrete[i] = (p, c / n)
        }
        Self { discrete }
    }
    fn pdf(&self, x: f64) -> f64 {
        if x < self.discrete[0].0 {
            return self.discrete[0].1
        }
        for (p1, p2) in self.discrete.iter().zip(self.discrete[1..].iter()) {
            if p1.0 <= x && x < p2.0 {
                return Self::cos_interp(x, *p1, *p2)
            }
        }
        self.discrete[self.discrete.len() - 1].1
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < self.discrete[0].0 {
            return x * self.discrete[0].1
        }
        let mut acc = self.discrete[0].0 * self.discrete[0].1;
        for (p1, p2) in self.discrete.iter().zip(self.discrete[1..].iter()) {
            if p1.0 <= x && x <= p2.0 {
                return acc + Self::cos_interp_int(x, *p1, *p2)
            } else if p2.0 < x {
                acc += (p1.1 + p2.1) / 2.0 * (p2.0 - p1.0)
            }
        }
        0.0 // error
    }
}
