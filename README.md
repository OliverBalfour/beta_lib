
This is a Rust library for computing credence calibration curves and Beta distributions. It comes with Python and JavaScript (WASM) bindings.

What's credence calibration? See [here](https://acritch.com/credence-game/) for details. Suppose you're forecasting future events, and you assign confidence scores like probabilities to each event. You might find that you're overconfident: maybe things you say happen with 70-80% confidence actually only happen 63% of the time on average. Thus, it's useful to be able to know the relationship between your confidence scores and the actual frequency.

The goal here is to be able to graph continuous curves of probability against confidence, with confidence intervals. It's not a simple regression problem because the outcomes are boolean but we want a continuous target. It's possible using a bit of probability theory, efficient and numerically stable pdf's for some common distributions, and some assumptions about how your confidence scores are distributed.

The Beta distribution functions in Python are at least 100x faster than `scipy.stats.beta`'s implementations (profiled informally).
