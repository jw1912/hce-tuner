use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::{data::NUM_PARAMS, DataPoint, Params, S};

pub struct Tuner {
    data: Vec<DataPoint>,
    weights: Params,
    momentum: Params,
    velocity: Params,
    threads: usize,
}

impl Tuner {
    pub fn new(threads: usize) -> Self {
        Self {
            data: Vec::new(),
            weights: Params::default(),
            momentum: Params::default(),
            velocity: Params::default(),
            threads,
        }
    }

    pub fn print_weights(&self) {
        for pc in 0..6 {
            println!("[");

            for rank in 0..8 {
                print!("        ");
                for file in 0..8 {
                    let idx = 64 * pc + 8 * rank + file;
                    print!("{:?},", self.weights[idx]);
                    if file != 7 {
                        print!(" ");
                    }
                }

                println!();
            }

            print!("    ], ");
        }
    }

    pub fn seed_weights(&mut self) {
        const VALS: [f64; 6] = [100.0, 300.0, 300.0, 500.0, 900.0, 0.0];

        for pc in 0..6 {
            let val = VALS[usize::from(pc)];
            let s = S(val, val);

            for sq in 0..64 {
                self.weights[64 * pc + sq] = s;
            }
        }
    }

    pub fn add_data(&mut self, file_path: &str) {
        let file = File::open(file_path).unwrap();
        let reader = BufReader::new(file);

        for line in reader.lines().map(|ln| ln.unwrap()) {
            let point = line.parse().unwrap();
            self.data.push(point);
        }
    }

    pub fn error(&self, k: f64) -> f64 {
        let chunk_size = (self.data.len() + self.threads - 1) / self.threads;
        let total_error = std::thread::scope(|s| {
            self.data
                .chunks(chunk_size)
                .map(|chunk| {
                    s.spawn(|| {
                        chunk
                            .iter()
                            .map(|point| point.error(k, &self.weights))
                            .sum::<f64>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|p| p.join().unwrap_or_default())
                .sum::<f64>()
        });

        (total_error / self.data.len() as f64) as f64
    }

    fn gradients(&self, k: f64) -> Params {
        let chunk_size = (self.data.len() + self.threads - 1) / self.threads;
        std::thread::scope(|s| {
            self.data
                .chunks(chunk_size)
                .map(|chunk| s.spawn(|| self.weights.gradients_batch(k, chunk)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|p| p.join().unwrap_or_default())
                .fold(Params::default(), |a, b| a + b)
        })
    }

    pub fn optimise_k(&self) -> f64 {
        let mut k = 0.009;
        let delta = 0.00001;
        let goal = 0.000001;
        let mut dev = 1f64;

        while dev.abs() > goal {
            let right = self.error(k + delta);
            let left = self.error(k - delta);
            dev = (right - left) / (5000. * delta);
            println!("k {k:.4} decr {left:.5} incr {right:.5}");
            k -= dev;
        }

        let error = self.error(k);
        println!("k {k:.6} error {error:.5}");

        k
    }

    pub fn run_epoch(&mut self, k: f64, rate: f64) {
        let gradients = self.gradients(k);
        const B1: f64 = 0.9;
        const B2: f64 = 0.999;

        for i in 0..NUM_PARAMS as u16 {
            let adj = (-2. * k / self.data.len() as f64) * gradients[i];
            self.momentum[i] = B1 * self.momentum[i] + (1. - B1) * adj;
            self.velocity[i] = B2 * self.velocity[i] + (1. - B2) * adj * adj;
            self.weights[i] -= rate * self.momentum[i] / (self.velocity[i].sqrt() + 0.00000001);
        }
    }
}
