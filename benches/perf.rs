use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("rand_large", |b| {
        let mut rng = black_box(rand::Rng::default());
        b.iter(|| {
            let mut sum = 0.0;
            for _ in 0..100_000 {
                sum += rng.rand();
            }
            black_box(sum);
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
