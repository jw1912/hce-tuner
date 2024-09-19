mod attacks;
mod data;
mod params;
mod score;
mod tuner;

use std::time::Instant;

use data::{DataPoint, NUM_PARAMS};
use params::Params;
use score::S;
use tuner::Tuner;

const THREADS: usize = 6;
const EPOCHS: usize = 5000;
const LRATE: f64 = 0.05;

fn main() {
    let mut tuner = Tuner::new(THREADS);

    tuner.seed_weights();
    tuner.add_data("gedas_filtered_sf_d9.epd");

    println!("Parameters: {NUM_PARAMS}");
    println!("Positions : {}", tuner.num_data_points());
    println!("Optimising k...");

    let k = tuner.optimise_k();

    println!("k = {k:.7}");

    let mut timer = Instant::now();

    for epoch in 1..=EPOCHS {
        tuner.run_epoch(k, LRATE);

        if epoch % 100 == 0 {
            let elapsed = timer.elapsed().as_secs_f64();
            let pps = (tuner.num_data_points() * 100) as f64 / elapsed;

            let error = tuner.error(k);
            println!("epoch {epoch} error {error:.5} time {elapsed:.2}s pos/sec {pps}");
            timer = Instant::now();
        }
    }

    tuner.print_weights();
}
