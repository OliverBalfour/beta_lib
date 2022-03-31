
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
    fn min_max(&self) -> (f64, f64);
    fn min(&self) -> f64 { self.min_max().0 }
    fn max(&self) -> f64 { self.min_max().1 }
}

impl Statistics for Vec<f64> {
    fn mean_variance(&self) -> (f64, f64) {
        let mut mean = 0.0;
        for val in self { mean += *val }
        mean /= self.len() as f64;
        let mut var = 0.0;
        for val in self {
            let z = *val - mean;
            var += z * z
        }
        var /= (self.len() - 1) as f64;
        (mean, var)
    }
    fn min_max(&self) -> (f64, f64) {
        let (mut min, mut max) = (std::f64::INFINITY, std::f64::NEG_INFINITY);
        for val in self {
            if *val < min { min = *val }
            if *val > max { max = *val }
        }
        (min, max)
    }
}
