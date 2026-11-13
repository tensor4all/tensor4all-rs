//! Large periodic-Gaussian transform without a dense grid or dense DFT.
//! Run in release mode; see the QFT guide for the numerical error contract.
#[path = "support/large_fourier.rs"]
mod large_fourier;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let bits = args.get(1).map_or(Ok(10), |x| x.parse())?;
    let dimensions = args.get(2).map_or(Ok(1), |x| x.parse())?;
    let interleaved = args.get(3).is_some_and(|x| x == "interleaved");
    // `frontend-only` skips the separately timed direct schedule.
    let runs = if args.get(4).is_some_and(|x| x == "frontend-only") {
        large_fourier::Runs::FrontEndOnly
    } else {
        large_fourier::Runs::ScheduleAndFrontEnd
    };
    large_fourier::run_case(bits, dimensions, interleaved, runs)?;
    Ok(())
}
