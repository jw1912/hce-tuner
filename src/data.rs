use std::str::FromStr;

use crate::{params::sigmoid, Params, S};

pub const NUM_PARAMS: usize = 384;
pub const TPHASE: f64 = 24.0;

#[derive(Default)]
pub struct DataPoint {
    pub active: [Vec<u16>; 2],
    pub phase: f64,
    pub result: f64,
}

impl DataPoint {
    pub fn eval(&self, params: &Params) -> f64 {
        let mut score = S::new(0.);

        for &idx in &self.active[0] {
            score += params[idx];
        }

        for &idx in &self.active[1] {
            score -= params[idx];
        }

        self.phase * score.0 + (1. - self.phase) * score.1
    }

    pub fn error(&self, k: f64, params: &Params) -> f64 {
        (self.result - sigmoid(k * self.eval(params))).powi(2)
    }
}

const CHARS: [char; 12] = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];
impl FromStr for DataPoint {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut pos = DataPoint::default();
        let mut row = 7;
        let mut col = 0;

        let mut parts = s.split(" ce ");
        let fen = parts.next().unwrap();
        let score = parts.next().unwrap();

        let mut in_stm = false;
        let mut stm = false;

        for ch in fen.chars() {
            if ch == '/' {
                row -= 1;
                col = 0;
            } else if in_stm {
                stm = ch == 'b';
                break;
            } else if ch == ' ' {
                in_stm = true;
            } else if ('1'..='8').contains(&ch) {
                col += ch.to_digit(10).expect("hard coded") as u16;
            } else if let Some(idx) = CHARS.iter().position(|&element| element == ch) {
                let c = idx / 6;
                let pc = idx as u16 - 6 * c as u16;
                let sq = 8 * row + col;
                let flip = 56 * (c ^ 1) as u16;

                pos.active[c].push(pc * 64 + (sq ^ flip));
                pos.phase += [0., 1., 1., 2., 4., 0.][pc as usize];

                col += 1;
            }
        }

        if pos.phase > TPHASE {
            pos.phase = TPHASE
        }

        pos.phase /= TPHASE;

        pos.result = score.parse().unwrap();

        if stm {
            pos.result = 1.0 - pos.result;
        }

        Ok(pos)
    }
}
