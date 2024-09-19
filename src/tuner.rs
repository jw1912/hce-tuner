use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::{data::{Offset, NUM_PARAMS}, DataPoint, Params, S};

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

    pub fn num_data_points(&self) -> usize {
        self.data.len()
    }

    pub fn print_weights(&self) {
        println!();

        let print_pst = |start: u16, tabs, cols, rows| {
            for rank in 0..rows {
                print!("{tabs}");
                for file in 0..cols {
                    let idx = start + cols * rank + file;
                    print!("{:?},", self.weights[idx]);
                    if file != cols - 1 {
                        print!(" ");
                    }
                }

                println!();
            }
        };

        println!("static PST: [[S; 64]; 8] = [[S(0, 0); 64], [S(0, 0); 64],");
        for pc in 0..6 {
            println!("[");
            print_pst(Offset::PST + 64 * pc, "        ", 8, 8);
            print!("    ], ");
        }

        println!();

        println!("];");

        let print_simple_table = |name, size, start: u16| {
            println!();
            print!("const {name}: [S; {size}] = [");

            for file in 0..size {
                print!("{:?}", self.weights[start + file]);

                if file != size - 1 {
                    print!(", ");
                }
            }

            println!("];");
        };

        print_simple_table("ROOK_SEMI_OPEN_FILE", 8, Offset::SEMI_OPEN);

        print_simple_table("ROOK_FULL_OPEN_FILE", 8, Offset::FULL_OPEN);

        print_simple_table("ISOLATED_PAWN_FILE", 8, Offset::ISOLATED);

        println!();

        println!("static PASSED_PAWN_PST: [S; 64] = [");
        print_pst(Offset::PASSED, "    ", 8, 8);
        println!("];");

        println!();

        println!("static KNIGHT_MOBILITY: [S; 9] = [");
        print_pst(Offset::KNIGHT_MOBILITY, "    ", 3, 3);
        println!("];");

        println!();

        println!("static BISHOP_MOBILITY: [S; 14] = [");
        print_pst(Offset::BISHOP_MOBILITY, "    ", 7, 2);
        println!("];");

        println!();

        println!("static ROOK_MOBILITY: [S; 15] = [");
        print_pst(Offset::ROOK_MOBILITY, "    ", 5, 3);
        println!("];");

        println!();

        println!("static QUEEN_MOBILITY: [S; 28] = [");
        print_pst(Offset::QUEEN_MOBILITY, "    ", 7, 4);
        println!("];");

        println!();

        println!("const BISHOP_PAIR: S = {:?};", self.weights[Offset::BISHOP_PAIR])
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
            k -= dev;

            if k <= 0.0 {
                println!("k {k:.4} decr {left:.5} incr {right:.5}");
            }
        }

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
