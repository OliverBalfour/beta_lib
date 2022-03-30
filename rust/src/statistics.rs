
// Like statrs::statistics::Statistics except the methods
// don't move the iterator
pub trait Statistics {
    fn mean(&self) -> f64 {
        self.mean_variance().0
    }
    // Sample variance and std dev
    fn variance(&self) -> f64 {
        self.mean_variance().1
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    fn mean_variance(&self) -> (f64, f64) {
        (self.mean(), self.variance())
    }
}

impl Statistics for Vec<f64> {
    fn mean_variance(&self) -> (f64, f64) {
        let mut mean = 0.0;
        for val in self { mean += *val }
        let mut var = 0.0;
        for val in self {
            let z = *val - mean;
            var += z * z
        }
        var /= self.len() as f64 - 1.0;
        (mean, var)
    }
}
