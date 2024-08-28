use criterion::{black_box, criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use std::time::Instant;

// Function to benchmark
fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand_benchmark");

    // Expected threshold in nanoseconds (e.g., 50 ns)
    let default_median_time_ns = 10;
    let expected_median_time_ns = match std::env::var("BENCHMARK_RAND_MEDIAN_TIME_NS") {
        Ok(v) => match v.parse::<u128>() {
            Ok(v) => v,
            Err(_why) => default_median_time_ns,
        },
        Err(_why) => default_median_time_ns,
    };

    group.bench_function("rand", |b: &mut Bencher| {
        b.iter_custom(|iters| {
            let mut rng = black_box(rand::Rng::default());
            let start = Instant::now();
            for _ in 0..iters {
                let x = rng.rand();
                black_box(x);
            }
            start.elapsed()
        });
    });

    // Add a custom check for performance regression
    let mut median_ave: f64 = 0.0;
    let mut median_count = 0;
    group.bench_with_input(BenchmarkId::new("check_regression", "rand"), &expected_median_time_ns, |b, &_threshold| {
        b.iter_custom(|iters| {
            let mut rng = black_box(rand::Rng::default());
            let start = Instant::now();
            for _ in 0..iters {
                let x = rng.rand();
                black_box(x);
            }
            let elapsed = start.elapsed();
            let median_time_ns = elapsed.as_nanos() / iters as u128;
            median_ave = (median_ave as f64 * median_count as f64 + median_time_ns as f64) / (median_count as f64 + 1.0) as f64;
            median_count += 1;

            elapsed
        });
    });

    println!("\nMedian {median_ave:.2} ns over {median_count} items.  Threshold set to {expected_median_time_ns} ns");

    // Assert that the median time per iteration is below the threshold
    assert!(
        median_ave <= expected_median_time_ns as f64,
        "Performance regression detected! Median time: {} ns exceeds threshold: {} ns",
        median_ave,
        expected_median_time_ns
    );

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
