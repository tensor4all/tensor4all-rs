use std::{error::Error, hint::black_box, time::Instant};
use tensor4all_quanticstransform::{
    quantics_fourier_operator, quantics_fourier_operator_multiaxis, AxisBit, FourierOptions,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let bits: usize = args[1].parse()?;
    let axes: usize = args[2].parse()?;
    let interleaved = args[3] == "interleaved";
    let layout: Vec<_> = if interleaved {
        (0..bits).flat_map(|bit| (0..axes).map(move |axis| AxisBit { axis, bit })).collect()
    } else {
        (0..axes).flat_map(|axis| (0..bits).map(move |bit| AxisBit { axis, bit })).collect()
    };
    let construct = || {
        if axes == 1 {
            quantics_fourier_operator(bits, FourierOptions::forward())
        } else {
            quantics_fourier_operator_multiaxis(&layout, FourierOptions::forward())
        }
    };
    let profile = construct()?.mpo().link_dims();
    let mut times = Vec::new();
    for _ in 0..15 {
        let start = Instant::now();
        let operator = black_box(construct()?);
        times.push(start.elapsed().as_secs_f64() * 1e3);
        assert_eq!(operator.mpo().link_dims(), profile);
    }
    times.sort_by(f64::total_cmp);
    println!("bits={bits} axes={axes} interleaved={interleaved} ms={:.4} bonds={profile:?}", times[7]);
    Ok(())
}
