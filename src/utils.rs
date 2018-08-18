/// Copied from https://github.com/bwasty/gltf-viewer/blob/master/src/utils.rs

use std::time::{ Instant, Duration };

pub fn elapsed(start_time: &Instant) -> String {
    let elapsed = start_time.elapsed();
    format_duration(elapsed)
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let nanos = duration.subsec_nanos();
    let ms = f64::from(nanos) / 1_000_000.0;
    if secs > 0 {
        let secs = secs as f64 + ms / 1000.0;
        format!("{:<4.*} s", 1, secs)
    }
    else {
        let places =
            if ms >= 20.0      { 0 }
            else if ms >= 1.0  { 1 }
            else {
                let micros = f64::from(nanos) / 1000.0;
                let places = if micros >= 10.0 { 0 } else { 2 };
                return format!("{:>3.*} µs", places, micros)
            };
        format!("{:>3.*} ms", places, ms)
    }
}

#[allow(dead_code)]
pub fn print_elapsed(message: &str, start_time: &Instant) {
    println!("{:<30}{}", message, elapsed(start_time));
}

pub fn print_elapsed_and_reset(message: &str, start_time: &mut Instant) {
    println!("{:<30}{}", message, elapsed(start_time));
    *start_time = Instant::now();
}
