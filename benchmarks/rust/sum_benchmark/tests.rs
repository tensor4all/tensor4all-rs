use super::*;
use serde_json::json;
use tensor4all_core::DynIndex;

fn config_value() -> Value {
    json!({"family":"fourier","size":2,"terms":4,"input_seed":0,"aci_seed":0,
        "method":"exact","tolerance":1e-10,"cutoff":1e-20,"prime_seconds":0.0})
}

fn known_config_value() -> Value {
    let mut value = config_value();
    value["family"] = json!("random");
    value["size"] = json!(3);
    value["terms"] = json!(2);
    value
}

#[test]
fn config_rejects_unbounded_or_ambiguous_inputs() {
    let value = config_value();
    let config = parse_config(value.clone()).unwrap();
    assert_eq!(config.original, value);
    for (key, invalid) in [
        ("family", json!("unknown")),
        ("method", json!("nary")),
        ("size", json!(0)),
        ("size", json!(7)),
        ("terms", json!(3)),
        ("input_seed", json!(-1)),
        ("input_seed", json!(u64::MAX)),
        ("aci_seed", json!(0.5)),
        ("tolerance", json!(0)),
        ("tolerance", json!(1.1)),
        ("cutoff", json!(-1e-20)),
        ("cutoff", json!(1.1)),
        ("prime_seconds", json!(-1)),
        ("prime_seconds", json!(61)),
        ("prime_seconds", json!("2")),
    ] {
        let mut bad = value.clone();
        bad[key] = invalid;
        assert!(parse_config(bad).is_err(), "accepted invalid {key}");
    }
    let mut missing = value.clone();
    missing.as_object_mut().unwrap().remove("cutoff");
    assert!(parse_config(missing).is_err());
    let mut extra = value;
    extra["max_bond_dim"] = json!(8);
    assert!(parse_config(extra).is_err());
    assert!(parse_config(json!([])).is_err());
    for family in ["smooth", "random", "mixed"] {
        let mut value = config_value();
        value["family"] = json!(family);
        value["size"] = json!(12);
        value["terms"] = json!(64);
        assert!(parse_config(value.clone()).is_ok());
        for (key, invalid) in [("size", 1), ("size", 13), ("terms", 0), ("terms", 65)] {
            let mut bad = value.clone();
            bad[key] = json!(invalid);
            assert!(parse_config(bad).is_err());
        }
    }
}

#[test]
fn dft_uses_normalization_reversed_input_bits_and_spectators() {
    let mut input = vec![C::new(0.0, 0.0); 8];
    input[1] = C::new(1.0, 0.0);
    input[6] = C::new(2.0, 0.0);
    let actual = inputs::normalized_dft(&input, 2).unwrap();
    let expected = [
        C::new(0.5, 0.0),
        C::new(-0.5, 0.0),
        C::new(0.5, 0.0),
        C::new(-0.5, 0.0),
        C::new(1.0, 0.0),
        C::new(0.0, -1.0),
        C::new(-1.0, 0.0),
        C::new(0.0, 1.0),
    ];
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (*a - b).norm())
        .fold(0.0, f64::max);
    assert!(error < 1e-14, "known DFT max error {error:e}");
    assert!(inputs::normalized_dft(&input[..7], 2).is_err());
    assert!(inputs::normalized_dft(&input, 0).is_err());
    assert!(inputs::normalized_dft(&input, 7).is_err());
    assert!(inputs::normalized_dft(&[], 2).is_err());
    assert!(inputs::normalized_dft(&vec![C::new(1.0, 0.0); 1024], 2).is_err());
    input[0] = C::new(f64::NAN, 0.0);
    assert!(inputs::normalized_dft(&input, 2).is_err());
}

fn known_case() -> inputs::Case {
    let sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let factors = [
        [[1.0, 2.0], [1.0, 3.0], [1.0, -1.0]],
        [[-1.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
    ];
    let terms = factors
        .into_iter()
        .map(|factors| {
            let bonds: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(1)).collect();
            let tensors = factors
                .into_iter()
                .zip(&sites)
                .enumerate()
                .map(|(j, (values, site))| {
                    let mut indices = Vec::new();
                    if j > 0 {
                        indices.push(bonds[j - 1].clone());
                    }
                    indices.push(site.clone());
                    if j < 2 {
                        indices.push(bonds[j].clone());
                    }
                    IdxTensor::from_dense(indices, values.map(|x| C::new(x, 0.0)).to_vec()).unwrap()
                })
                .collect();
            // Explicit dimension-one bonds keep the same connected chain as real fixtures.
            Net::from_tensors(tensors, (0..3).collect()).unwrap()
        })
        .collect();
    let values = [-1.0, 4.0, 2.0, 7.0, -5.0, 2.0, -5.0, -4.0];
    inputs::Case {
        terms,
        oracle: IdxTensor::from_dense(sites, values.map(|x| C::new(x, 0.0)).to_vec()).unwrap(),
        mask: vec![true, false, false, false, false, false, false, false],
    }
}

#[test]
fn all_six_methods_match_known_dense_sum() {
    let case = known_case();
    for method in [
        "aci",
        "sequential",
        "balanced",
        "density",
        "exact",
        "exact_truncate",
    ] {
        let mut value = known_config_value();
        value["method"] = json!(method);
        let config = parse_config(value).unwrap();
        let outcome = run_method(&config, &case.terms).unwrap();
        let result = report(&config, &case, outcome, 0.25).unwrap();
        let error = result["maxabs_error"].as_f64().unwrap();
        assert!(error < 1e-10, "{method} known-value error {error:e}");
        assert_eq!(result["schema_version"], 1);
        assert_eq!(result["config"], config.original);
        assert_eq!(result["seconds"], 0.25);
        assert_eq!(result["scalar"], "c64");
        assert_eq!(result["aci"].is_object(), method == "aci");
        assert_eq!(result["discarded_weight"].is_number(), method == "density");
        assert_eq!(result["reference"]["sum_re"], 0.0);
        assert_eq!(result["reference"]["sum_sq"], 140.0);
        assert_eq!(result["reference"]["maxabs"], 7.0);
        assert_eq!(result["cancellation_fraction"], 0.125);
    }
}

#[test]
fn fixture_oracles_and_seed_streams_are_reproducible() {
    for family in ["fourier", "smooth", "random", "mixed"] {
        let mut value = config_value();
        value["family"] = json!(family);
        value["terms"] = json!(if family == "fourier" { 4 } else { 3 });
        let config = parse_config(value.clone()).unwrap();
        let case = inputs::build_case(&config).unwrap();
        let repeated = inputs::build_case(&config).unwrap();
        let a = case.oracle.to_vec::<C>().unwrap();
        let b = repeated.oracle.to_vec::<C>().unwrap();
        assert_eq!(a, b, "{family} changed its deterministic stream");
        value["input_seed"] = json!(1);
        let other = inputs::build_case(&parse_config(value).unwrap()).unwrap();
        assert_ne!(
            a,
            other.oracle.to_vec::<C>().unwrap(),
            "{family} ignored input seed"
        );
        let outcome = run_method(&config, &case.terms).unwrap();
        let result = report(&config, &case, outcome, 0.25).unwrap();
        assert!(
            result["maxabs_relative_error"].as_f64().unwrap() < 1e-12,
            "{family}: {result}"
        );
    }
}

#[test]
fn original_1107_streams_match_known_dense_values() {
    for (family, values) in [
        (
            "smooth",
            vec![
                C::new(10.0, 0.0),
                C::new(1.0357249073240313, 0.0),
                C::new(1.7414705555311305, 0.0),
                C::new(0.6972342525239293, 0.0),
            ],
        ),
        (
            "mixed",
            vec![
                C::new(10.0, 0.0),
                C::new(1.9145258495562656, 0.0),
                C::new(3.0816355556605997, 0.0),
                C::new(1.2526738806059314, 0.0),
            ],
        ),
        (
            "random",
            vec![
                C::new(1.5931720409421202, -0.3854498285406248),
                C::new(-0.6571411659521723, 2.511103702441279),
                C::new(-1.3528466791857032, 1.63412124675101),
                C::new(-1.4233063215757344, 0.4014931900764102),
            ],
        ),
    ] {
        let mut value = config_value();
        value["family"] = json!(family);
        value["size"] = json!(2);
        value["terms"] = json!(1);
        let config = parse_config(value).unwrap();
        let case = inputs::build_case(&config).unwrap();
        let expected = IdxTensor::from_dense(case.oracle.indices().to_vec(), values).unwrap();
        let error = case.oracle.sub(&expected).unwrap().maxabs().unwrap();
        assert!(
            error < 1e-12,
            "#1107 {family} original stream changed: {error:e}"
        );
    }
}

#[test]
fn reporting_rejects_nonfinite_timing_and_checksums() {
    let config = parse_config(known_config_value()).unwrap();
    let case = known_case();
    for seconds in [f64::NAN, f64::INFINITY, -1.0] {
        let outcome = run_method(&config, &case.terms).unwrap();
        assert!(report(&config, &case, outcome, seconds).is_err());
    }
    for value in [
        C::new(f64::NAN, 0.0),
        C::new(0.0, f64::INFINITY),
        C::new(1e308, 0.0),
    ] {
        assert!(checksum(&[value]).is_err());
    }
}

#[test]
fn metrics_use_the_oracle_scale_and_cancellation_mask() {
    let config = parse_config(known_config_value()).unwrap();
    let mut case = known_case();
    let outcome = run_method(&config, &case.terms).unwrap();
    let values = [-2.0, 4.0, 2.0, 7.0, -5.0, 2.0, -5.0, -6.0];
    case.oracle = IdxTensor::from_dense(
        case.oracle.indices().to_vec(),
        values.map(|x| C::new(x, 0.0)).to_vec(),
    )
    .unwrap();
    let result = report(&config, &case, outcome, 0.25).unwrap();
    for (field, expected) in [
        ("maxabs_error", 2.0),
        ("relative_error", (5.0f64 / 163.0).sqrt()),
        ("maxabs_relative_error", 2.0 / 7.0),
        ("cancellation_maxabs_relative_error", 1.0 / 7.0),
    ] {
        let actual = result[field].as_f64().unwrap();
        assert!(
            (actual - expected).abs() < 1e-14,
            "{field}: {actual} != {expected}"
        );
    }
    assert_eq!(result["checksum"]["weighted_re"], -39.0 / 8.0);
    assert_eq!(result["reference"]["sum_sq"], 163.0);
}

#[test]
fn exact_preserves_direct_sum_rank_and_truncating_methods_compress() {
    let sites: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let bond = DynIndex::new_dyn(1);
    // A basis state gives an exact zero density tail even at the declared cutoff.
    let values = vec![C::new(1.0, 0.0), C::new(0.0, 0.0)];
    let tensors = vec![
        IdxTensor::from_dense(vec![sites[0].clone(), bond.clone()], values.clone()).unwrap(),
        IdxTensor::from_dense(vec![bond, sites[1].clone()], values).unwrap(),
    ];
    let term = Net::from_tensors(tensors, vec![0, 1]).unwrap();
    let terms = vec![term.clone(), term];
    for method in [
        "aci",
        "sequential",
        "balanced",
        "density",
        "exact",
        "exact_truncate",
    ] {
        let mut value = known_config_value();
        value["size"] = json!(2);
        value["method"] = json!(method);
        let config = parse_config(value).unwrap();
        let outcome = run_method(&config, &terms).unwrap();
        let expected = if method == "exact" { 2 } else { 1 };
        assert_eq!(outcome.network.link_dims(), vec![expected], "{method}");
    }
}
