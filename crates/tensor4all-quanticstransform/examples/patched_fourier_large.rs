//! Large periodic-Gaussian transform without a dense grid or dense DFT.
//! Run in release mode; see the QFT guide for the numerical error contract.
#[path = "support/large_fourier.rs"]
mod large_fourier;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let bits = args.get(1).map_or(Ok(10), |x| x.parse())?;
    let dimensions = args.get(2).map_or(Ok(1), |x| x.parse())?;
    let interleaved = args.get(3).is_some_and(|x| x == "interleaved");
    // `frontend-only` skips the separately timed direct schedule, and
    // `certified=<share>` applies the operator with a certified truncation
    // (#1286) given that share of the allowance.
    let runs = if args.iter().skip(4).any(|x| x == "frontend-only") {
        large_fourier::Runs::FrontEndOnly
    } else {
        large_fourier::Runs::ScheduleAndFrontEnd
    };
    let certified_share = args
        .iter()
        .skip(4)
        .find_map(|x| x.strip_prefix("certified="))
        .map(str::parse)
        .transpose()?;
    large_fourier::run_case(bits, dimensions, interleaved, runs, certified_share)?;
    Ok(())
}
