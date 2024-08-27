use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("rand", |b| {
        let mut rng = black_box(rand::Rng::default());
        b.iter(|| {
            let x = rng.rand();
            black_box(x);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
