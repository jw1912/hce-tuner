use std::str::FromStr;

use crate::{params::sigmoid, Params, S};

pub const NUM_PARAMS: usize = 384 + 8;
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

        let mut occ = [0; 2];
        let mut bbs = [[0; 6]; 2];

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

                let bit = 1u64 << sq;
                bbs[c][usize::from(pc)] |= bit;
                occ[c] |= bit;

                pos.phase += [0., 1., 1., 2., 4., 0.][pc as usize];

                col += 1;
            }
        }

        for side in [0, 1] {
            let ksq = bbs[side][5].trailing_zeros();

            let cflip = if side == 0 { 56 } else { 0 };
            let kflip = if ksq % 8 > 3 { 7 } else { 0 };
            let flip = cflip ^ kflip;

            for piece in 0..6 {
                let mut bb = bbs[side][piece];

                while bb > 0 {
                    let sq = bb.trailing_zeros() as u16;
                    let fsq = sq ^ flip;

                    pos.active[side].push(64 * piece as u16 + fsq);

                    // rooks
                    if piece == 3 && (0x101010101010101 << (sq % 8)) & bbs[side][0] == 0 {
                        pos.active[side].push(384 + (fsq % 8));
                    }

                    bb &= bb - 1;
                }
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
