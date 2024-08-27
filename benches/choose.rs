use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("choose_0_100", |b| {
        let mut rng = black_box(rand::Rng::default());
        let range = black_box((0..=100).collect::<Vec<_>>());
        b.iter(|| {
            let x = rng.choose(range.iter());
            black_box(x)
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
