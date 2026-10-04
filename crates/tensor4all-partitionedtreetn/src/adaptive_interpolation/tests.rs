//! Unit tests of the private helpers of the patch driver.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};

use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{NodeNameNetwork, TreeTN};

use super::cache::{
    is_finite, value_defect, Counters, Entries, KeyLayout, OutOfRange, PackedKey, PatchCache,
    PatchSampler, ValueDefect, WordMap,
};
use super::embed::{check_outcome_layout, embed_fixed_sites, exact_active_network};
use super::layout::SiteLayout;
use super::sampling::{
    all_points, build_candidates, mix64, patch_candidates, patch_seeds, point_list_capacity,
    uniform_points, PatchDomain, SplitMix64,
};
use super::{active_pivots, added_pivots, complete_points, PatchedInterpolationOptions};
use crate::{PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN};

// ---------------------------------------------------------------------------
// SplitMix64, Lemire, and seeds. The expected values were computed with an
// independent Python implementation of the documented mapping.
// ---------------------------------------------------------------------------

#[test]
fn splitmix64_matches_the_reference_stream() {
    let mut rng = SplitMix64::new(1_234_567);
    let stream: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
    assert_eq!(
        stream,
        [
            6_457_827_717_110_365_317,
            3_203_168_211_198_807_973,
            9_817_491_932_198_370_423,
            4_593_380_528_125_082_431,
            16_408_922_859_458_223_821,
        ]
    );
    assert_eq!(mix64(1_234_567), stream[0]);
}

#[test]
fn lemire_draws_follow_the_documented_mapping() {
    let mut rng = SplitMix64::new(2024);
    let draws: Vec<u64> = [3, 5, 7, 1, 10].iter().map(|&d| rng.below(d)).collect();
    assert_eq!(draws, [1, 0, 2, 0, 8]);

    // A bound just above 2^63 rejects about half of the raw draws; these four
    // values need five rejections.
    let mut rng = SplitMix64::new(7);
    let bound = (1u64 << 63) + 1;
    let draws: Vec<u64> = (0..4).map(|_| rng.below(bound)).collect();
    assert_eq!(
        draws,
        [
            3_595_544_800_446_187_243,
            8_308_050_873_407_804_673,
            2_300_599_727_732_774_152,
            1_238_314_238_945_538_992,
        ]
    );
    assert!(draws.iter().all(|&draw| draw < bound));
}

#[test]
fn patch_seeds_mix_the_root_seed_with_the_path() {
    let root = patch_seeds(0, &[]);
    assert_eq!(root.candidates, 5_341_624_319_751_574_814);
    assert_eq!(root.engine, 17_577_709_759_210_329_795);
    let child = patch_seeds(0, &[(2, 1), (0, 3)]);
    assert_eq!(child.candidates, 9_896_170_195_978_033_743);
    assert_eq!(child.engine, 8_337_709_925_750_577_402);

    // Every pair component, the pair order, and the root seed matter.
    let variants = [
        patch_seeds(1, &[(2, 1), (0, 3)]),
        patch_seeds(0, &[(2, 1), (0, 2)]),
        patch_seeds(0, &[(1, 1), (0, 3)]),
        patch_seeds(0, &[(0, 3), (2, 1)]),
        patch_seeds(0, &[(2, 1)]),
    ];
    let distinct: HashSet<u64> = variants
        .iter()
        .chain([&root, &child])
        .flat_map(|seeds| [seeds.candidates, seeds.engine])
        .collect();
    assert_eq!(distinct.len(), 2 * (variants.len() + 2));
}

#[test]
fn measurement_streams_follow_the_documented_mapping() {
    // Expected values from an independent Python implementation of
    // SplitMix64, the path state, and the stream selectors.
    let root = patch_seeds(0, &[]);
    assert_eq!(root.zero_screen(), 13_481_018_753_965_960_313);
    assert_eq!(root.verify(0), 10_784_900_928_564_024_043);
    assert_eq!(root.verify(1), 1_731_796_998_791_982_555);
    assert_eq!(root.audit(), 3_965_753_011_800_562_021);
    assert_eq!(root.scale(), 11_195_191_274_241_726_103);
    assert_eq!(root.engine_run(0), root.engine);
    assert_eq!(root.engine_run(1), 14_625_715_455_650_629_719);
    let child = patch_seeds(0, &[(2, 1), (0, 3)]);
    assert_eq!(child.zero_screen(), 1_272_130_066_354_424_498);
    assert_eq!(child.verify(0), 977_359_825_929_394_526);
    assert_eq!(child.verify(1), 18_371_716_087_452_245_731);
    assert_eq!(child.audit(), 3_762_845_222_878_257_995);
    assert_eq!(child.scale(), 7_950_964_838_826_632_979);
    assert_eq!(child.engine_run(1), 15_846_517_643_923_597_972);
    // Sampled points draw every coordinate in active-site order.
    assert_eq!(
        columns(&uniform_points(&[3, 5, 7], 4, 42).unwrap(), 3),
        [vec![2, 0, 1], vec![1, 0, 6], vec![0, 4, 2], vec![1, 1, 3]]
    );
    // Exhaustive points are column-major, first active site fastest.
    assert_eq!(
        columns(&all_points(&[2, 3], 6).unwrap(), 2),
        [
            vec![0, 0],
            vec![1, 0],
            vec![0, 1],
            vec![1, 1],
            vec![0, 2],
            vec![1, 2]
        ]
    );
}

#[test]
fn point_lists_reject_lengths_that_cannot_fit_a_vec_allocation() {
    assert_eq!(point_list_capacity(1, usize::MAX), None);
    assert!(all_points(&[2], usize::MAX).is_err());
    assert!(uniform_points(&[2], usize::MAX, 7).is_err());
}

fn domain_parts(dims: &[usize], fixed: &[Option<usize>]) -> (Vec<usize>, KeyLayout) {
    let active: Vec<usize> = (0..dims.len()).filter(|&p| fixed[p].is_none()).collect();
    let layout = KeyLayout::new(active.iter().map(|&p| dims[p]).collect());
    (active, layout)
}

fn no_pivots(n_sites: usize) -> ColMajorArray<usize> {
    ColMajorArray::new(vec![], vec![n_sites, 0]).unwrap()
}

fn columns(points: &[usize], n_active: usize) -> Vec<Vec<usize>> {
    points.chunks(n_active).map(<[usize]>::to_vec).collect()
}

#[test]
fn random_candidates_are_a_fixed_list_for_a_fixed_seed() {
    let dims = [3, 5, 7];
    let fixed = [None; 3];
    let (active, layout) = domain_parts(&dims, &fixed);
    let domain = PatchDomain {
        dims: &dims,
        fixed: &fixed,
        active: &active,
        layout: &layout,
    };
    let candidates = patch_candidates(&domain, &no_pivots(3), &[], 4, 42);
    assert_eq!(candidates.count, 4);
    assert_eq!(
        columns(&candidates.points, 3),
        [vec![2, 0, 1], vec![1, 0, 6], vec![0, 4, 2], vec![1, 1, 3]]
    );
}

#[test]
fn candidates_keep_compatible_user_then_recycled_pivots_in_order() {
    // Site 1 is fixed to 2; active sites are 0 and 2.
    let dims = [3, 4, 2];
    let fixed = [None, Some(2), None];
    let (active, layout) = domain_parts(&dims, &fixed);
    let domain = PatchDomain {
        dims: &dims,
        fixed: &fixed,
        active: &active,
        layout: &layout,
    };
    let user = ColMajorArray::new(
        vec![
            2, 2, 1, // compatible -> (2, 1)
            0, 1, 0, // incompatible (site 1 is 1)
            2, 2, 1, // duplicate of the first
            0, 2, 0, // compatible -> (0, 0)
        ],
        vec![3, 4],
    )
    .unwrap();
    let recycled = vec![vec![1, 2, 1], vec![0, 2, 0], vec![1, 3, 1]];
    // Target 2 is already met by the user pivots, but recycled pivots are kept.
    let candidates = patch_candidates(&domain, &user, &recycled, 2, 9);
    assert_eq!(
        columns(&candidates.points, 2),
        [vec![2, 1], vec![0, 0], vec![1, 1]]
    );
}

#[test]
fn candidates_fall_back_to_column_major_points_and_stop_at_the_patch_size() {
    let dims = [2, 3];
    let fixed = [None, None];
    let (active, layout) = domain_parts(&dims, &fixed);
    let domain = PatchDomain {
        dims: &dims,
        fixed: &fixed,
        active: &active,
        layout: &layout,
    };
    let user = ColMajorArray::new(vec![1, 0], vec![2, 1]).unwrap();
    // No random attempts: the first unused points in column-major order.
    let candidates = build_candidates(&domain, &user, &[], 4, 0, |_| 0);
    assert_eq!(
        columns(&candidates.points, 2),
        [vec![1, 0], vec![0, 0], vec![0, 1], vec![1, 1]]
    );
    // A target above the patch size yields every point once.
    let candidates = patch_candidates(&domain, &user, &[], 100, 3);
    assert_eq!(candidates.count, 6);
    let distinct: HashSet<Vec<usize>> = columns(&candidates.points, 2).into_iter().collect();
    assert_eq!(distinct.len(), 6);
}

// ---------------------------------------------------------------------------
// Packed keys and the patch cache.
// ---------------------------------------------------------------------------

#[test]
fn keys_pack_coordinates_without_straddling_words() {
    // Three 43-bit coordinates: no two fit in one word.
    let wide = KeyLayout::new(vec![1usize << 43; 3]);
    assert_eq!(wide.n_words(), 3);
    let point = [(1usize << 43) - 1, 5, (1usize << 42) + 3];
    assert_eq!(wide.decode(&wide.encode(point.iter().copied())), point);

    // 129 binary sites need three words; points differing only in the last
    // site have different keys.
    let binary = KeyLayout::new(vec![2; 129]);
    assert_eq!(binary.n_words(), 3);
    let mut last = vec![0usize; 129];
    let first = binary.encode(last.iter().copied());
    last[128] = 1;
    let second = binary.encode(last.iter().copied());
    assert_ne!(first, second);
    assert_eq!(binary.decode(&second), last);

    // A dimension-one site takes no bits; an empty layout has an empty key.
    let unit = KeyLayout::new(vec![1, 4, 1]);
    assert_eq!(unit.n_words(), 1);
    assert_eq!(unit.decode(&unit.encode([0, 3, 0].into_iter())), [0, 3, 0]);
    let empty = KeyLayout::new(vec![]);
    assert_eq!(empty.n_words(), 0);
    assert!(empty.encode(std::iter::empty()).is_empty());

    // A full 64-bit coordinate.
    let full = KeyLayout::new(vec![usize::MAX, 3]);
    assert_eq!(full.n_words(), 2);
    let point = [usize::MAX - 1, 2];
    assert_eq!(full.decode(&full.encode(point.iter().copied())), point);
}

#[test]
fn cache_split_hands_every_entry_to_its_child() {
    let mut cache = PatchCache::new(vec![2, 3, 2]);
    for a in 0..2 {
        for b in 0..3 {
            for c in 0..2 {
                let key = cache.layout().encode([a, b, c].into_iter());
                cache.insert(key, (100 * a + 10 * b + c) as f64);
            }
        }
    }
    let children = cache.split(1);
    assert_eq!(children.len(), 3);
    for (b, child) in children.iter().enumerate() {
        assert_eq!(child.len(), 4);
        assert_eq!(child.layout().dims(), [2, 2]);
        for a in 0..2 {
            for c in 0..2 {
                let key = child.layout().encode([a, c].into_iter());
                assert_eq!(child.get(&key), Some((100 * a + 10 * b + c) as f64));
            }
        }
    }
}

#[test]
fn largest_cached_points_are_ordered_by_magnitude_then_coordinates() {
    // Values by point on a 2 x 3 layout: two ties at magnitude 5, one zero.
    let mut cache = PatchCache::new(vec![2, 3]);
    let values = [
        ([0, 0], 1.0),
        ([1, 0], -5.0),
        ([0, 1], 5.0),
        ([1, 1], 0.0),
        ([0, 2], -7.0),
        ([1, 2], 2.0),
    ];
    for (point, value) in values {
        cache.insert(cache.layout().encode(point.into_iter()), value);
    }
    let magnitude = |value: f64| value.abs();
    // The tie at 5 is ordered by coordinates: [0, 1] before [1, 0].
    assert_eq!(
        cache.largest_points(3, magnitude),
        [vec![0, 2], vec![0, 1], vec![1, 0]]
    );
    // The zero entry is never returned, even with room left.
    assert_eq!(
        cache.largest_points(10, magnitude),
        [vec![0, 2], vec![0, 1], vec![1, 0], vec![1, 2], vec![0, 0]]
    );
    assert!(cache.largest_points(0, magnitude).is_empty());

    // The same answer for every key type.
    for dims in [vec![1 << 40, 1 << 40], vec![1 << 43; 3]] {
        let mut cache = PatchCache::new(dims.clone());
        let far: Vec<usize> = dims.iter().map(|&dim| dim - 1).collect();
        let origin = vec![0usize; dims.len()];
        let mut middle = origin.clone();
        middle[0] = 1;
        for (point, value) in [(&far, 3.0), (&origin, -3.0), (&middle, 4.0)] {
            cache.insert(cache.layout().encode(point.iter().copied()), value);
        }
        assert_eq!(
            cache.largest_points(2, magnitude),
            [middle.clone(), origin.clone()],
            "{dims:?}"
        );
    }
}

/// The key type a cache uses: 1 (`u64`), 2 (`u128`), or 3 (boxed slice).
fn key_kind<T: Copy>(cache: &PatchCache<T>) -> usize {
    match cache.entries() {
        Entries::Narrow(_) => 1,
        Entries::Double(_) => 2,
        Entries::Wide(_) => 3,
    }
}

#[test]
fn cache_key_type_follows_the_word_count() {
    // An empty layout and one word use the inline u64 key, two words the
    // inline u128 key, and three or more words the boxed key.
    let cases: [(Vec<usize>, usize, usize); 5] = [
        (vec![], 0, 1),
        (vec![2, 3], 1, 1),
        (vec![1 << 40, 1 << 40], 2, 2),
        (vec![1 << 43; 3], 3, 3),
        (vec![2; 129], 3, 3),
    ];
    for (dims, n_words, kind) in cases {
        let cache = PatchCache::<f64>::new(dims.clone());
        assert_eq!(cache.layout().n_words(), n_words, "{dims:?}");
        assert_eq!(key_kind(&cache), kind, "{dims:?}");
    }
}

#[test]
fn borrowed_lookups_find_owned_keys_of_every_key_type() {
    fn check<K: PackedKey>(words: &[u64], other: &[u64]) {
        let mut map: WordMap<K, f64> = WordMap::default();
        map.insert(K::from_words(words), 1.5);
        assert_eq!(K::find(&map, words), Some(&1.5));
        assert_eq!(K::find(&map, other), None);
        for (index, &word) in words.iter().enumerate() {
            assert_eq!(K::from_words(words).word(index), word);
        }
    }
    check::<u64>(&[0b1011], &[0b1010]);
    check::<u128>(&[7, u64::MAX], &[7, 0]);
    check::<Box<[u64]>>(&[1, 2, 3], &[1, 2, 4]);

    // A key encoded into a reused buffer finds the entry stored under the
    // owned key of the same point, for every key type.
    for dims in [vec![2, 3, 2], vec![1 << 40, 1 << 40], vec![1 << 43; 3]] {
        let mut cache = PatchCache::new(dims.clone());
        let point: Vec<usize> = dims.iter().map(|&dim| dim - 1).collect();
        cache.insert(cache.layout().encode(point.iter().copied()), 4.0);
        let mut buffer = vec![0u64; cache.layout().n_words()];
        cache.layout().encode_checked(&point, &mut buffer).unwrap();
        assert_eq!(cache.get(&buffer), Some(4.0), "{dims:?}");
        let origin = vec![0usize; dims.len()];
        cache.layout().encode_checked(&origin, &mut buffer).unwrap();
        assert_eq!(cache.get(&buffer), None, "{dims:?}");
    }
}

#[test]
fn checked_encoding_matches_encode_and_rejects_out_of_range_coordinates() {
    let layout = KeyLayout::new(vec![1 << 43, 5, 1 << 43]);
    let point = [(1usize << 43) - 1, 4, 17];
    let mut buffer = vec![u64::MAX; layout.n_words()];
    layout.encode_checked(&point, &mut buffer).unwrap();
    assert_eq!(*buffer, *layout.encode(point.iter().copied()));
    assert_eq!(
        layout.encode_checked(&[0, 5, 0], &mut buffer),
        Err(OutOfRange {
            slot: 1,
            value: 5,
            dim: 5
        })
    );
}

#[test]
fn cache_split_moves_keys_without_re_encoding() {
    // Three 21-bit sites fit one word; removing the middle one keeps the
    // other two at their parent word and shift.
    let dims = [3usize, 1 << 20, 2];
    let value = |p: &[usize]| (p[0] + 10 * p[1] + 100 * p[2]) as f64;
    let mut cache = PatchCache::new(dims.to_vec());
    let points: Vec<[usize; 3]> = (0..3)
        .flat_map(|a| [(a, 0), (a, 1)])
        .flat_map(|(a, c)| [[a, 0, c], [a, (1 << 20) - 1, c], [a, 12_345, c]])
        .collect();
    for point in &points {
        cache.insert(cache.layout().encode(point.iter().copied()), value(point));
    }
    let parent_layout = cache.layout().clone();
    let children = cache.split(0);
    assert_eq!(children.len(), 3);
    for (a, child) in children.iter().enumerate() {
        assert_eq!(child.layout().dims(), [1 << 20, 2]);
        assert_eq!(child.layout().n_words(), 1);
        assert_eq!(key_kind(child), 1);
        assert_eq!(child.len(), 6);
        for point in points.iter().filter(|point| point[0] == a) {
            // The child key is the parent key with the split coordinate's
            // bits cleared.
            let parent_key = parent_layout.encode(point.iter().copied());
            let cleared = parent_layout.encode([0, point[1], point[2]].into_iter());
            let child_key = child.layout().encode([point[1], point[2]].into_iter());
            assert_eq!(child_key, cleared);
            assert_eq!(parent_key == cleared, a == 0);
            assert_eq!(child.get(&child_key), Some(value(point)));
            assert_eq!(child.layout().decode(&child_key), [point[1], point[2]]);
        }
    }

    // Splitting a moved-key child again keeps its entries reachable.
    let grandchildren = children.into_iter().nth(2).unwrap().split(1);
    for (c, grandchild) in grandchildren.iter().enumerate() {
        assert_eq!(grandchild.len(), 3);
        for b in [0, (1 << 20) - 1, 12_345] {
            let key = grandchild.layout().encode([b].into_iter());
            assert_eq!(grandchild.get(&key), Some(value(&[2, b, c])));
        }
    }
}

/// Every point of the grid whose coordinate at slot `i` ranges over
/// `choices[i]`, first slot fastest.
fn grid_points(choices: &[Vec<usize>]) -> Vec<Vec<usize>> {
    choices
        .iter()
        .fold(vec![Vec::new()], |points, slot_choices| {
            slot_choices
                .iter()
                .flat_map(|&value| {
                    points.iter().map(move |point| {
                        let mut point = point.clone();
                        point.push(value);
                        point
                    })
                })
                .collect()
        })
}

/// Fill a cache over `dims` with the grid of `choices` (value: the point's
/// index plus one half), split it at `slot`, and check that every child has
/// the expected key type, keeps the parent's word count, holds exactly the
/// points with its coordinate at `slot`, and returns each point's value.
fn check_masked_split(dims: &[usize], choices: &[Vec<usize>], slot: usize, kind: usize) {
    let points = grid_points(choices);
    let mut cache = PatchCache::new(dims.to_vec());
    for (index, point) in points.iter().enumerate() {
        cache.insert(
            cache.layout().encode(point.iter().copied()),
            index as f64 + 0.5,
        );
    }
    assert_eq!(cache.len(), points.len(), "the grid points are distinct");
    assert_eq!(key_kind(&cache), kind);
    let n_words = cache.layout().n_words();
    let children = cache.split(slot);
    assert_eq!(children.len(), dims[slot]);
    for (coordinate, child) in children.iter().enumerate() {
        assert_eq!(child.layout().n_words(), n_words, "keys are moved");
        assert_eq!(key_kind(child), kind);
        let inside: Vec<(usize, &Vec<usize>)> = points
            .iter()
            .enumerate()
            .filter(|(_, point)| point[slot] == coordinate)
            .collect();
        assert_eq!(child.len(), inside.len(), "child {coordinate}");
        for (index, point) in inside {
            let mut rest = point.clone();
            rest.remove(slot);
            let key = child.layout().encode(rest.iter().copied());
            assert_eq!(child.get(&key), Some(index as f64 + 0.5), "{point:?}");
        }
    }
}

#[test]
fn cache_split_moves_two_word_keys_at_a_slot_in_the_second_word() {
    // Words: [a (40 bits)], [b (40 bits), c (2 bits, shift 40), d (1 bit)].
    // Removing c leaves two words, so the keys are moved, not re-encoded.
    let big = 1usize << 40;
    let layout = KeyLayout::new(vec![big, big, 3, 2]);
    assert_eq!(layout.n_words(), 2);
    check_masked_split(
        &[big, big, 3, 2],
        &[
            vec![0, 1, big - 1],
            vec![0, 5, big - 1],
            vec![0, 1, 2],
            vec![0, 1],
        ],
        2,
        2,
    );
}

#[test]
fn cache_split_moves_wide_keys_at_a_slot_past_the_first_word() {
    // Words: [a (43 bits)], [b (43 bits), c (2 bits, shift 43)], [d (43
    // bits)]. Removing c leaves three words, so the boxed keys are moved.
    let big = 1usize << 43;
    let layout = KeyLayout::new(vec![big, big, 3, big]);
    assert_eq!(layout.n_words(), 3);
    check_masked_split(
        &[big, big, 3, big],
        &[
            vec![0, 1, big - 1],
            vec![0, 7, big - 1],
            vec![0, 1, 2],
            vec![0, big - 1],
        ],
        2,
        3,
    );
}

#[test]
fn cache_split_re_encodes_only_when_a_word_is_freed() {
    // 129 binary sites need three words; 128 fit two, so the children are
    // re-encoded into the inline u128 key. Their children keep two words and
    // move their keys.
    let points: Vec<Vec<usize>> = (0..40u64)
        .map(|seed| {
            (0..129u64)
                .map(|site| (mix64(seed * 1000 + site) & 1) as usize)
                .collect()
        })
        .collect();
    let distinct: HashSet<&Vec<usize>> = points.iter().collect();
    assert_eq!(distinct.len(), points.len());
    let value = |index: usize| index as f64 + 0.5;
    let mut cache = PatchCache::new(vec![2usize; 129]);
    for (index, point) in points.iter().enumerate() {
        cache.insert(cache.layout().encode(point.iter().copied()), value(index));
    }
    assert_eq!(cache.len(), points.len());
    assert_eq!(key_kind(&cache), 3);
    let children = cache.split(64);
    for (coordinate, child) in children.iter().enumerate() {
        assert_eq!(child.layout().n_words(), 2);
        assert_eq!(key_kind(child), 2);
        let inside: Vec<usize> = (0..points.len())
            .filter(|&index| points[index][64] == coordinate)
            .collect();
        assert!(!inside.is_empty());
        assert_eq!(child.len(), inside.len());
        for index in inside {
            let mut rest = points[index].clone();
            rest.remove(64);
            let key = child.layout().encode(rest.iter().copied());
            assert_eq!(child.get(&key), Some(value(index)));
        }
    }

    // The grandchildren of the first child (site 64 fixed to 0) split at the
    // first remaining site, moving their two-word keys.
    let grandchildren = children.into_iter().next().unwrap().split(0);
    for (coordinate, grandchild) in grandchildren.iter().enumerate() {
        assert_eq!(grandchild.layout().n_words(), 2);
        assert_eq!(key_kind(grandchild), 2);
        let inside: Vec<usize> = (0..points.len())
            .filter(|&index| points[index][64] == 0 && points[index][0] == coordinate)
            .collect();
        assert!(!inside.is_empty());
        assert_eq!(grandchild.len(), inside.len());
        for index in inside {
            let mut rest = points[index].clone();
            rest.remove(64);
            rest.remove(0);
            let key = grandchild.layout().encode(rest.iter().copied());
            assert_eq!(grandchild.get(&key), Some(value(index)));
        }
    }
}

// ---------------------------------------------------------------------------
// The patch sampler.
// ---------------------------------------------------------------------------

type BoxedEvaluator = Box<dyn Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>>>;

/// Run `body` with a sampler for a three-site patch whose middle site is
/// fixed to 1; return its result with the evaluation and cache-hit counts.
fn with_sampler<R>(
    evaluate: BoxedEvaluator,
    body: impl FnOnce(&PatchSampler<'_, f64, BoxedEvaluator>) -> R,
) -> (R, usize, usize) {
    let counters = Counters::default();
    let fixed = [None, Some(1), None];
    let sampler = PatchSampler {
        evaluate: &evaluate,
        fixed: &fixed,
        n_active: 2,
        counters: &counters,
        cache: RefCell::new(PatchCache::new(vec![2, 3])),
    };
    let result = body(&sampler);
    assert_eq!(sampler.cache.borrow().len(), counters.evaluations.get());
    (
        result,
        counters.evaluations.get(),
        counters.cache_hits.get(),
    )
}

fn batch_values(
    sampler: &PatchSampler<'_, f64, BoxedEvaluator>,
    data: &[usize],
) -> anyhow::Result<Vec<f64>> {
    let shape = [2, data.len() / 2];
    sampler.sample(ColMajorArrayRef::new(data, &shape)?)
}

#[test]
fn sampler_evaluates_each_new_point_once_with_fixed_coordinates() {
    let seen = std::rc::Rc::new(RefCell::new(Vec::new()));
    let log = seen.clone();
    let evaluate: BoxedEvaluator = Box::new(move |batch| {
        let points: Vec<Vec<usize>> = batch.data().chunks(3).map(<[usize]>::to_vec).collect();
        log.borrow_mut().extend(points.clone());
        Ok(points
            .iter()
            .map(|p| (100 * p[0] + 10 * p[1] + p[2]) as f64)
            .collect())
    });
    let (values, evaluations, hits) = with_sampler(evaluate, |sampler| {
        let first = batch_values(sampler, &[0, 0, 1, 2, 0, 0]).unwrap();
        let second = batch_values(sampler, &[1, 2, 1, 1, 0, 0]).unwrap();
        (first, second)
    });
    assert_eq!(values.0, [10.0, 112.0, 10.0]);
    assert_eq!(values.1, [112.0, 111.0, 10.0]);
    assert_eq!(
        *seen.borrow(),
        [vec![0, 1, 0], vec![1, 1, 2], vec![1, 1, 1]]
    );
    assert_eq!(evaluations, 3);
    // One duplicate inside the first batch and two cached points in the second.
    assert_eq!(hits, 3);
}

#[test]
fn sampler_rejects_invalid_batches_and_evaluator_results_without_caching() {
    let constant: fn() -> BoxedEvaluator = || Box::new(|batch| Ok(vec![1.0; batch.shape()[1]]));
    let cases: Vec<(BoxedEvaluator, Vec<usize>, Vec<usize>, &str)> = vec![
        (
            constant(),
            vec![0, 0, 0],
            vec![3, 1],
            "does not match the 2 active",
        ),
        (
            constant(),
            vec![0, 3],
            vec![2, 1],
            "out of range for dimension 3",
        ),
        (
            Box::new(|_| Ok(vec![1.0])),
            vec![0, 0, 1, 1],
            vec![2, 2],
            "returned 1 values for 2 points",
        ),
        (
            Box::new(|batch| Ok(vec![f64::NAN; batch.shape()[1]])),
            vec![1, 2],
            vec![2, 1],
            "non-finite value at the point [1, 1, 2]",
        ),
        (
            Box::new(|batch| Ok(vec![f64::INFINITY; batch.shape()[1]])),
            vec![1, 2],
            vec![2, 1],
            "non-finite",
        ),
        (
            Box::new(|_| Err(anyhow::anyhow!("user failure"))),
            vec![1, 2],
            vec![2, 1],
            "user failure",
        ),
    ];
    for (evaluate, data, shape, needle) in cases {
        let (error, evaluations, _) = with_sampler(evaluate, |sampler| {
            sampler
                .sample(ColMajorArrayRef::new(&data, &shape).unwrap())
                .unwrap_err()
        });
        assert!(error.to_string().contains(needle), "{error} lacks {needle}");
        assert_eq!(evaluations, 0);
    }
}

#[test]
fn sampler_gives_the_same_values_and_counts_for_every_key_type() {
    // One fixed site (coordinate 1) before the active sites; the value
    // weights every active coordinate by its position.
    let value = |active: &[usize]| {
        active
            .iter()
            .enumerate()
            .map(|(i, &v)| (i + 1) as f64 * v as f64)
            .sum::<f64>()
    };
    // The first active site is small, as the split below creates one child
    // per coordinate. Splitting the two-word layout frees a word (re-encoded
    // into a u64 key); the others move their keys.
    for (dims, kind) in [
        (vec![2, 3], 1),
        (vec![3, 1 << 63], 2),
        (vec![3, 1 << 43, 1 << 43, 1 << 43], 3),
    ] {
        let n_active = dims.len();
        let corner = |pick: usize| -> Vec<usize> {
            dims.iter()
                .enumerate()
                .map(|(i, &dim)| [0, 1, dim - 1][(pick + i) % 3])
                .collect()
        };
        let (p1, p2, p3, p4) = (corner(0), corner(1), corner(2), vec![1; n_active]);
        let counters = Counters::default();
        let mut fixed = vec![Some(1)];
        fixed.extend(std::iter::repeat_n(None, n_active));
        let evaluate: BoxedEvaluator = Box::new(move |batch| {
            let rows = batch.shape()[0];
            Ok(batch
                .data()
                .chunks(rows)
                .map(|point| {
                    assert_eq!(point[0], 1);
                    value(&point[1..])
                })
                .collect())
        });
        let sampler = PatchSampler {
            evaluate: &evaluate,
            fixed: &fixed,
            n_active,
            counters: &counters,
            cache: RefCell::new(PatchCache::new(dims.clone())),
        };
        assert_eq!(key_kind(&sampler.cache.borrow()), kind);
        let sample = |points: &[&Vec<usize>]| {
            let data: Vec<usize> = points.iter().flat_map(|p| p.iter().copied()).collect();
            let shape = [n_active, points.len()];
            sampler
                .sample(ColMajorArrayRef::new(&data, &shape).unwrap())
                .unwrap()
        };
        assert_eq!(
            sample(&[&p1, &p2, &p1, &p3]),
            [value(&p1), value(&p2), value(&p1), value(&p3)]
        );
        assert_eq!(
            sample(&[&p2, &p4, &p4]),
            [value(&p2), value(&p4), value(&p4)]
        );
        assert_eq!(counters.evaluations.get(), 4, "{dims:?}");
        assert_eq!(counters.cache_hits.get(), 3, "{dims:?}");
        assert_eq!(sampler.cache.borrow().len(), 4);

        // After a split at the first active site, the child that contains
        // `p1` serves it from the moved entries: a hit, no evaluation.
        let children = sampler.cache.into_inner().split(0);
        let child = children.into_iter().nth(p1[0]).unwrap();
        let mut child_fixed = fixed.clone();
        child_fixed[1] = Some(p1[0]);
        let child_sampler = PatchSampler {
            evaluate: &evaluate,
            fixed: &child_fixed,
            n_active: n_active - 1,
            counters: &counters,
            cache: RefCell::new(child),
        };
        let rest = &p1[1..];
        let values = child_sampler
            .sample(ColMajorArrayRef::new(rest, &[n_active - 1, 1]).unwrap())
            .unwrap();
        assert_eq!(values, [value(&p1)]);
        assert_eq!(counters.evaluations.get(), 4, "{dims:?}");
        assert_eq!(counters.cache_hits.get(), 4, "{dims:?}");
    }
}

// ---------------------------------------------------------------------------
// Layout, exact networks, re-embedding, and outcome checks.
// ---------------------------------------------------------------------------

/// Chain 0 - 1 - 2; node 0 has sites a (2) and b (3), node 1 none, node 2 c (2).
fn chain_layout() -> (SiteLayout<usize>, Vec<DynIndex>) {
    let sites = vec![
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(3),
        DynIndex::new_dyn(2),
    ];
    let mut topology = NodeNameNetwork::new();
    for node in 0..3usize {
        topology.add_node(node).unwrap();
    }
    topology.add_edge(&0, &1).unwrap();
    topology.add_edge(&1, &2).unwrap();
    let node_sites = BTreeMap::from([
        (0usize, vec![sites[0].clone(), sites[1].clone()]),
        (1, vec![]),
        (2, vec![sites[2].clone()]),
    ]);
    let layout = SiteLayout::validated(
        topology,
        node_sites,
        &no_pivots(3),
        &PatchedInterpolationOptions::new(2).with_error_norm(crate::ErrorNorm::sampled_max()),
    )
    .unwrap();
    (layout, sites)
}

/// Dense values of `network` in the column-major order of `sites`.
fn dense_values(network: &TreeTN<IdxTensor, usize>, sites: &[DynIndex]) -> Vec<f64> {
    let zero = IdxTensor::from_dense(sites.to_vec(), vec![0.0; 12]).unwrap();
    // Adding the network to an explicit zero aligns the axes with `sites`.
    zero.add(&network.to_dense().unwrap())
        .unwrap()
        .to_vec::<f64>()
        .unwrap()
}

#[test]
fn exact_network_without_active_site_sits_on_the_smallest_node() {
    let (layout, sites) = chain_layout();
    let fixed = [Some(1), Some(2), Some(0)];
    let network = exact_active_network(&layout, &[], vec![7.5_f64]).unwrap();
    assert!(network.link_dims().iter().all(|&dim| dim == 1));
    let node0 = network.node_index(&0).unwrap();
    assert_eq!(
        network.tensor(node0).unwrap().to_vec::<f64>().unwrap(),
        [7.5]
    );
    let embedded = embed_fixed_sites::<f64, usize>(&network, &layout, &fixed).unwrap();
    // Only (a, b, c) = (1, 2, 0) is nonzero: column-major position 1 + 2 * 2.
    let mut expected = vec![0.0; 12];
    expected[5] = 7.5;
    assert_eq!(dense_values(&embedded, &sites), expected);
}

#[test]
fn exact_network_rejects_more_than_one_active_site() {
    let (layout, _) = chain_layout();
    let error = exact_active_network(&layout, &[0, 2], vec![1.0_f64; 4]).unwrap_err();
    assert!(matches!(error, PartitionedTreeTNError::TreeTN { .. }));
}

/// Build an exact network with site c active and a, b fixed, re-embed it, and
/// check that every node tensor (values, ones, and one-hot factors) has the
/// scalar type `T`.
fn assert_embedding_keeps_dtype<T>(value: T, is_dtype: fn(&IdxTensor) -> bool)
where
    T: tensor4all_core::CommonScalar + tensor4all_core::TensorElement,
{
    let (layout, _) = chain_layout();
    for (active, fixed) in [
        (vec![2], [Some(1), Some(2), None]),
        (vec![], [Some(0), Some(1), Some(1)]),
    ] {
        let values = vec![value; if active.is_empty() { 1 } else { 2 }];
        let network = exact_active_network(&layout, &active, values).unwrap();
        let embedded = embed_fixed_sites::<T, usize>(&network, &layout, &fixed).unwrap();
        assert_eq!(embedded.node_count(), 3);
        for name in embedded.node_names() {
            let tensor = embedded
                .tensor(embedded.node_index(&name).unwrap())
                .unwrap();
            assert!(is_dtype(tensor), "node {name} has another dtype");
        }
    }
}

#[test]
fn exact_and_embedded_networks_keep_the_scalar_type() {
    assert_embedding_keeps_dtype(2.0_f32, IdxTensor::is_f32);
    assert_embedding_keeps_dtype(2.0_f64, IdxTensor::is_f64);
    assert_embedding_keeps_dtype(num_complex::Complex32::new(1.0, 2.0), IdxTensor::is_c32);
    assert_embedding_keeps_dtype(num_complex::Complex64::new(1.0, 2.0), IdxTensor::is_c64);
}

/// Finiteness and magnitude checks for one real type and its complex type:
/// infinities of both signs and NaN in each component are non-finite;
/// subnormals, `-0.0`, and the largest finite values are finite; a complex
/// value with finite parts whose magnitude overflows is its own defect.
macro_rules! value_defect_cases {
    ($name:ident, $real:ty, $complex:ty) => {
        #[test]
        fn $name() {
            let subnormal = <$real>::MIN_POSITIVE / 4.0;
            assert!(subnormal.is_subnormal());
            let finite = [
                0.0,
                -0.0,
                1.0,
                -2.5,
                subnormal,
                -subnormal,
                <$real>::MAX,
                <$real>::MIN,
            ];
            let bad = [<$real>::INFINITY, <$real>::NEG_INFINITY, <$real>::NAN];
            for x in finite {
                assert_eq!(value_defect(x), None, "{x:?}");
                assert_eq!(value_defect(<$complex>::new(x, 0.0)), None, "{x:?}");
                assert_eq!(value_defect(<$complex>::new(-0.0, x)), None, "{x:?}");
            }
            assert_eq!(value_defect(<$complex>::new(subnormal, -subnormal)), None);
            for x in bad {
                assert_eq!(value_defect(x), Some(ValueDefect::NonFinite), "{x:?}");
                for value in [
                    <$complex>::new(x, 1.0),
                    <$complex>::new(1.0, x),
                    <$complex>::new(x, x),
                ] {
                    assert!(!is_finite(value), "{value:?}");
                    assert_eq!(
                        value_defect(value),
                        Some(ValueDefect::NonFinite),
                        "{value:?}"
                    );
                }
            }
            // Finite parts, overflowing magnitude.
            for value in [
                <$complex>::new(<$real>::MAX, <$real>::MAX),
                <$complex>::new(<$real>::MIN, <$real>::MAX / 1.5),
            ] {
                assert!(is_finite(value), "{value:?}");
                assert_eq!(
                    value_defect(value),
                    Some(ValueDefect::MagnitudeOverflow),
                    "{value:?}"
                );
            }
            // A real value never overflows its magnitude.
            assert_eq!(value_defect(<$real>::MIN), None);
        }
    };
}

value_defect_cases!(
    value_defects_of_f32_and_complex32,
    f32,
    num_complex::Complex32
);
value_defect_cases!(
    value_defects_of_f64_and_complex64,
    f64,
    num_complex::Complex64
);

#[test]
fn outcome_layout_check_rejects_every_mismatch() {
    let (layout, sites) = chain_layout();
    let link01 = DynIndex::new_dyn(1);
    let link12 = DynIndex::new_dyn(1);
    let tensor = |indices: Vec<DynIndex>| {
        let size = indices.iter().map(IndexLike::dim).product();
        IdxTensor::from_dense(indices, vec![1.0_f64; size]).unwrap()
    };
    let chain = |node0: Vec<DynIndex>, node2: Vec<DynIndex>, names: Vec<usize>| {
        TreeTN::from_tensors(
            vec![
                tensor([node0, vec![link01.clone()]].concat()),
                tensor(vec![link01.clone(), link12.clone()]),
                tensor([vec![link12.clone()], node2].concat()),
            ],
            names,
        )
        .unwrap()
    };
    let fixed = [None, Some(0), None];
    let matching = chain(
        vec![sites[0].clone()],
        vec![sites[2].clone()],
        vec![0, 1, 2],
    );
    assert!(check_outcome_layout(&matching, &layout, &fixed).is_ok());

    // A missing active site, a different identity, and a different dimension.
    let missing = chain(vec![sites[0].clone()], vec![], vec![0, 1, 2]);
    assert!(check_outcome_layout(&missing, &layout, &fixed)
        .unwrap_err()
        .contains("node 2"));
    let renamed = chain(vec![sites[0].sim()], vec![sites[2].clone()], vec![0, 1, 2]);
    assert!(check_outcome_layout(&renamed, &layout, &fixed)
        .unwrap_err()
        .contains("node 0"));
    let mut resized_site = sites[2].clone();
    resized_site.dim = 3;
    let resized = chain(vec![sites[0].clone()], vec![resized_site], vec![0, 1, 2]);
    assert!(check_outcome_layout(&resized, &layout, &fixed)
        .unwrap_err()
        .contains("node 2"));

    // Too few nodes, a renamed node, and a different edge set.
    let two_nodes = TreeTN::from_tensors(
        vec![
            tensor(vec![sites[0].clone(), link01.clone()]),
            tensor(vec![link01.clone(), sites[2].clone()]),
        ],
        vec![0usize, 2],
    )
    .unwrap();
    assert!(check_outcome_layout(&two_nodes, &layout, &fixed)
        .unwrap_err()
        .contains("has 2 nodes"));
    let renamed_node = chain(
        vec![sites[0].clone()],
        vec![sites[2].clone()],
        vec![0, 1, 5],
    );
    assert!(check_outcome_layout(&renamed_node, &layout, &fixed)
        .unwrap_err()
        .contains("no node 2"));
    let star = TreeTN::from_tensors(
        vec![
            tensor(vec![sites[0].clone(), link01.clone(), link12.clone()]),
            tensor(vec![link01.clone()]),
            tensor(vec![link12.clone(), sites[2].clone()]),
        ],
        vec![0usize, 1, 2],
    )
    .unwrap();
    assert!(check_outcome_layout(&star, &layout, &fixed)
        .unwrap_err()
        .contains("lacks the edge"));
}

#[test]
fn completed_pivots_carry_the_fixed_coordinates() {
    let dims = [2, 3, 4];
    let fixed = [None, Some(2), None];
    let active = [0, 2];
    assert!(active_pivots(None, &active, &dims).unwrap().is_empty());
    let pivots = ColMajorArray::new(vec![1, 3, 0, 0], vec![2, 2]).unwrap();
    let local = active_pivots(Some(&pivots), &active, &dims).unwrap();
    assert_eq!(local, [vec![1, 3], vec![0, 0]]);
    assert_eq!(
        complete_points(&local, &fixed),
        [vec![1, 2, 3], vec![0, 2, 0]]
    );
    let one_d = ColMajorArray::new(vec![1, 3], vec![2]).unwrap();
    assert!(active_pivots(Some(&one_d), &active, &dims)
        .unwrap_err()
        .contains("expected a 2D array"));
    let rows = ColMajorArray::new(vec![1, 1, 1], vec![3, 1]).unwrap();
    assert!(active_pivots(Some(&rows), &active, &dims)
        .unwrap_err()
        .contains("3 rows"));
    let range = ColMajorArray::new(vec![1, 4], vec![2, 1]).unwrap();
    assert!(active_pivots(Some(&range), &active, &dims)
        .unwrap_err()
        .contains("coordinate 4"));
}

#[test]
fn added_pivots_put_worst_points_first_and_drop_known_points() {
    let base: HashSet<Vec<usize>> = [vec![0, 0], vec![1, 1]].into_iter().collect();
    let worst = [vec![2, 2], vec![1, 1], vec![3, 3]];
    let outcome = [vec![3, 3], vec![0, 0], vec![4, 4], vec![5, 5]];
    // Base candidates and repeats are dropped; the list stops at the limit.
    assert_eq!(
        added_pivots(&base, &worst, &outcome, 3),
        [vec![2, 2], vec![3, 3], vec![4, 4]]
    );
    assert_eq!(
        added_pivots(&base, &worst, &outcome, 10),
        [vec![2, 2], vec![3, 3], vec![4, 4], vec![5, 5]]
    );
    assert!(added_pivots(&base, &[], &[], 3).is_empty());
}

// ---------------------------------------------------------------------------
// Assembly of disjoint patches.
// ---------------------------------------------------------------------------

fn one_site_patch<T: tensor4all_core::TensorElement>(
    site: &DynIndex,
    values: Vec<T>,
    coordinate: usize,
) -> SubDomainTreeTN<usize> {
    named_one_site_patch(0, site, values, coordinate)
}

fn named_one_site_patch<T: tensor4all_core::TensorElement>(
    node: usize,
    site: &DynIndex,
    values: Vec<T>,
    coordinate: usize,
) -> SubDomainTreeTN<usize> {
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], values).unwrap()],
        vec![node],
    )
    .unwrap();
    SubDomainTreeTN::new(
        tree,
        Projector::from_pairs([(site.clone(), coordinate)]).unwrap(),
    )
    .unwrap()
}

#[test]
fn disjoint_assembly_checks_structure_and_dtype() {
    let site = DynIndex::new_dyn(2);
    let empty = PartitionedTreeTN::<usize>::from_disjoint_subdomains(vec![]).unwrap();
    assert!(empty.is_empty());

    let partition = PartitionedTreeTN::from_disjoint_subdomains(vec![
        one_site_patch(&site, vec![1.0_f64, 0.0], 0),
        one_site_patch(&site, vec![0.0_f64, 2.0], 1),
    ])
    .unwrap();
    assert_eq!(partition.len(), 2);

    let other = DynIndex::new_dyn(2);
    let error = PartitionedTreeTN::from_disjoint_subdomains(vec![
        one_site_patch(&site, vec![1.0_f64, 0.0], 0),
        one_site_patch(&other, vec![0.0_f64, 2.0], 1),
    ])
    .unwrap_err();
    assert!(matches!(error, PartitionedTreeTNError::SiteIndexMismatch));

    let error = PartitionedTreeTN::from_disjoint_subdomains(vec![
        one_site_patch(&site, vec![1.0_f64, 0.0], 0),
        one_site_patch(&site, vec![num_complex::Complex64::new(0.0, 0.0); 2], 1),
    ])
    .unwrap_err();
    assert!(matches!(
        error,
        PartitionedTreeTNError::DTypeMismatch { .. }
    ));

    let error = PartitionedTreeTN::from_disjoint_subdomains(vec![
        named_one_site_patch(0, &site, vec![1.0_f64, 0.0], 0),
        named_one_site_patch(1, &site, vec![0.0_f64, 2.0], 1),
    ])
    .unwrap_err();
    assert!(matches!(error, PartitionedTreeTNError::TopologyMismatch));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "pairwise disjoint projectors")]
fn disjoint_assembly_asserts_disjointness_in_debug_builds() {
    let site = DynIndex::new_dyn(2);
    let _ = PartitionedTreeTN::from_disjoint_subdomains(vec![
        one_site_patch(&site, vec![1.0_f64, 0.0], 0),
        one_site_patch(&site, vec![1.0_f64, 0.0], 0),
    ]);
}
