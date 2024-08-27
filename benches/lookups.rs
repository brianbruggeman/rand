use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("get_fast_1d_noise", |b| {
        b.iter(|| {
            let position: u32 = black_box(12345_u32);
            let seed: u32 = black_box(67890_u32);

            let x: u32 = rand::get_fast_1d_noise(position, seed);
            black_box(x)
        })
    });

    c.bench_function("rand_from", |b| {
        let rng = black_box(rand::Rng::default());
        b.iter(|| {
            let x = rng.rand_from(black_box(0.0));
            black_box(x)
        })
    });

    c.bench_function("random_from", |b| {
        let rng = black_box(rand::Rng::default());
        b.iter(|| {
            let x = rng.random_from::<u32>(black_box(0.0));
            black_box(x)
        })
    });

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
