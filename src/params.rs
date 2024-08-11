use crate::data::DataPoint;

use super::{NUM_PARAMS, S};
use std::ops::*;

pub struct Params([S; NUM_PARAMS]);

impl Default for Params {
    fn default() -> Self {
        Params([S::new(0.); NUM_PARAMS])
    }
}

impl Index<u16> for Params {
    type Output = S;
    fn index(&self, index: u16) -> &Self::Output {
        &self.0[usize::from(index)]
    }
}

impl IndexMut<u16> for Params {
    fn index_mut(&mut self, index: u16) -> &mut Self::Output {
        &mut self.0[usize::from(index)]
    }
}

impl Add<Params> for Params {
    type Output = Params;
    fn add(mut self, rhs: Params) -> Self::Output {
        for i in 0..NUM_PARAMS {
            self.0[i] += rhs.0[i];
        }
        self
    }
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

impl Params {
    pub fn gradients_batch(&self, k: f64, data: &[DataPoint]) -> Self {
        let mut grad = Params::default();

        for point in data {
            let sigm = sigmoid(k * point.eval(self));
            let term = (point.result - sigm) * (1. - sigm) * sigm;
            let phase_adj = term * S(point.phase, 1. - point.phase);

            for &idx in &point.active[0] {
                grad[idx] += phase_adj;
            }

            for &idx in &point.active[1] {
                grad[idx] -= phase_adj;
            }
        }

        grad
    }
}
