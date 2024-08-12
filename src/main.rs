mod data;
mod params;
mod score;
mod tuner;

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

    let k = tuner.optimise_k();

    for epoch in 1..=EPOCHS {
        tuner.run_epoch(k, LRATE);

        if epoch % 100 == 0 {
            let error = tuner.error(k);
            println!("epoch {epoch} error {error:.5}");
        }
    }

    tuner.print_weights();
}
