use bellman::pairing::ff::*;
use bellman::pairing::bn256::Fr;
use bellman::plonk::domains::Domain;

#[test]
fn try_16_bits() {
    const WIDTH: usize = 16;

    let max = 1u64 << WIDTH;

    let domain = Domain::<Fr>::new_for_size((WIDTH * 2) as u64).unwrap();
    let generator = domain.generator;

    let mut gen_powers = [Fr::zero(); WIDTH * 2];
    let mut map: std::collections::HashMap::<Fr, (u16, usize)> = std::collections::HashMap::new();

    let mut current = Fr::one();
    gen_powers[0] = current;
    for idx in 1..(WIDTH*2) {
        current.mul_assign(&generator);
        gen_powers[idx] = current;
    }

    current.mul_assign(&generator);
    assert_eq!(current, Fr::one());

    for value in 1..max {
        let value = value as u16;
        let encoding = encode(value, &gen_powers);
        for shift in 1..WIDTH {
            let mut shifted = encoding;
            shifted.mul_assign(&gen_powers[shift]);

            if let Some(val) = map.get(&shifted) {
                let this_rotated_value = rotate_left(value as u16, shift);
                let other_rotated_value = rotate_left(val.0, val.1);
                if this_rotated_value != other_rotated_value {
                    panic!("Same encoding of {} for value {:#016b}, shift {} and value {:#016b}, shift {}", shifted, val.0, val.1, value, shift);
                }
            } else {
                map.insert(shifted, (value, shift));
            }
        }
    }
}

fn get_16_bits(mut value: u16) -> [bool; 16] {
    let mut result = [false; 16];
    for idx in 0..16 {
        result[idx] = value & 1 == 1;
        value >>= 1;
    }

    result
}

fn get_32_bits(mut value: u32) -> [bool; 32] {
    let mut result = [false; 32];
    for idx in 0..32 {
        result[idx] = value & 1 == 1;
        value >>= 1;
    }

    result
}

fn rotate_left(value: u16, rotation: usize) -> u16 {
    if rotation == 0 {
        value
    } else {
        value << rotation | value >> (16 - rotation)
    }
}

fn rotate_left_32(value: u32, rotation: usize) -> u32 {
    if rotation == 0 {
        value
    } else {
        value << rotation | value >> (32 - rotation)
    }
}

fn encode(value: u16, gen_powers: &[Fr]) -> Fr {
    let bits = get_16_bits(value);
    let mut encoding = Fr::zero();
    for (&b, g) in bits.iter().zip(gen_powers.iter()) {
        if b {
            encoding.add_assign(&g);
        }
    }

    encoding
}

fn encode_32(value: u32, gen_powers: &[Fr]) -> Fr {
    let bits = get_32_bits(value);
    let mut encoding = Fr::zero();
    for (&b, g) in bits.iter().zip(gen_powers.iter()) {
        if b {
            encoding.add_assign(&g);
        }
    }

    encoding
}

#[test]
fn multicore_try_32_bits() {
    const WIDTH: usize = 32;
    const WIDTH_ENCODING: usize = 5;

    let worker = bellman::worker::Worker::new();
    let max = 1u64 << WIDTH;

    let domain = Domain::<Fr>::new_for_size((WIDTH * 2) as u64).unwrap();
    let generator = domain.generator;

    let mut gen_powers = [Fr::zero(); WIDTH * 2];

    let mut current = Fr::one();
    gen_powers[0] = current;
    for idx in 1..(WIDTH*2) {
        current.mul_assign(&generator);
        gen_powers[idx] = current;
    }

    current.mul_assign(&generator);
    assert_eq!(current, Fr::one());

    let mut results = Vec::with_capacity(max as usize);
    unsafe {results.set_len(max as usize)};
    // let mut results = vec![Fr::zero(); (max as usize) * WIDTH];
    let len = results.len();
    println!("Start working for {:x} elements", len);
    worker.scope(len, |scope, chunk_size| {
        let mut start_idx = 0;
        for chunk in results.chunks_mut(chunk_size) {
            let start = start_idx as u64;
            scope.spawn(move |_| {
                let mut idx = start;
                for el in chunk.iter_mut() {
                    *el = encode_32(idx as u32, &gen_powers);
                    idx += 1;
                }
            });
            start_idx += chunk_size;
        }
    });

    println!("Finished pre-generation");

    // first trivial checks: there is no trivial encoding as is

    // let mut set = std::collections::HashSet::with_capacity(1 << WIDTH);
    // for shift_1 in 0..WIDTH {
    //     set.clear();

    //     let mul_by = gen_powers[shift_1 as usize];
    //     // insert initial
    //     for (idx, &el) in results.iter().enumerate() {
    //         let mut el = el;
    //         el.mul_assign(&mul_by);
    //         if set.contains(&el) {
    //             panic!("explicit duplicate at shift {}: element {} for encoding of {:#032b}", shift_1, el, idx);
    //         } else {
    //             set.insert(el);
    //         }
    //     }

    //     println!("Finished checking that shift {} contains no duplicates", shift_1);
    // }

    // drop(set);

    println!("Allocate sets");

    let num_threads = worker.get_num_spawned_threads(1 << WIDTH);
    // let mut sets = vec![std::collections::HashSet::with_capacity((1 << WIDTH) / num_threads); num_threads];
    let mut sets = vec![std::collections::HashSet::new(); num_threads];
    // let mut sets = vec![std::collections::BTreeSet::new(); num_threads];

    let initial_capacity = sets[0].capacity();

    println!("Start verification of encodings and shifts for uniqueness in them");
    for shift_1 in 0..WIDTH {
        let start = std::time::Instant::now();
        // insert initial
        let mul_by = gen_powers[shift_1 as usize];
        worker.scope(len, |scope, chunk_size| {
            let mut start_idx = 0;
            for (chunk, set) in results.chunks(chunk_size).zip(sets.chunks_mut(1)) {
                scope.spawn(move |_| {
                    let set = &mut set[0];
                    set.clear();
                    if set.capacity() == initial_capacity {
                        set.reserve((1 << WIDTH) / num_threads);
                    }
                    let mut idx = start_idx;
                    for e in chunk.iter() {
                        let mut el = *e;
                        el.mul_assign(&mul_by);

                        // let el = el.into_repr();

                        if set.contains(&el) {
                            panic!("explicit duplicate at shift {}: element {} for encoding of {:#032b}", shift_1, el, idx);
                        } else {
                            set.insert(el);
                        }
                        idx += 1;
                    }
                });

                start_idx += chunk_size;
            }
        });

        println!("Insertions taken {:?}", start.elapsed());
        let start = std::time::Instant::now();

        let sets = &sets;
        worker.scope(len, |scope, chunk_size| {
            let mut start_idx = 0;
            for (chunk_idx, chunk) in results.chunks(chunk_size).enumerate() {
                scope.spawn(move |_| {
                    for set_idx in 0..sets.len() {
                        if set_idx == chunk_idx {
                            continue;
                        }
                        let set = &sets[set_idx];
                        let this_set = &sets[chunk_idx];
                        if !this_set.is_disjoint(set) {
                            panic!("encoding is not unique!");
                        }
                        // let mut idx = start_idx;
                        // for el in chunk.iter() {
                        //     let el = el.into_repr();
                        //     if set.contains(&el) {
                        //         panic!("explicit duplicate at shift {}: element {} for encoding of {:#032b}", shift_1, el, idx);
                        //     }
                        //     idx += 1;
                        // }
                    }
                });

                start_idx += chunk_size;
            }
        });

        println!("Verification taken {:?}", start.elapsed());

        println!("Finished checking that shift {} contains no duplicates", shift_1);
    }

    drop(sets);

    // set's are unique in themselves, so now need to check
    // across checks
    println!("Start uniquenss checks across checks");
    let mut maps: Vec<std::collections::HashMap::<Fr, (u32, u8)>> = vec![std::collections::HashMap::new(); num_threads];
    let initial_capacity = maps[0].capacity();

    for shift_1 in 0..WIDTH {
        for shift_2 in (shift_1+1)..WIDTH {
            let start = std::time::Instant::now();
            // insert initial maps for values of shift_1
            let mul_by = gen_powers[shift_1 as usize];
            worker.scope(len, |scope, chunk_size| {
                let mut start_idx = 0;
                for (chunk, map) in results.chunks(chunk_size).zip(maps.chunks_mut(1)) {
                    scope.spawn(move |_| {
                        let map = &mut map[0];
                        map.clear();
                        if map.capacity() == initial_capacity {
                            map.reserve((1 << WIDTH) / num_threads);
                        }
                        let mut idx = start_idx;
                        for e in chunk.iter() {
                            let mut el = *e;
                            el.mul_assign(&mul_by);

                            map.insert(el, (idx as u32, shift_1 as u8));
                            idx += 1;
                        }
                    });

                    start_idx += chunk_size;
                }
            });

            println!("Insertions taken {:?}", start.elapsed());
            let start = std::time::Instant::now();

            let maps_ref = &maps;
            // check
            let mul_by = gen_powers[shift_2 as usize];
            worker.scope(len, |scope, chunk_size| {
                let mut start_idx = 0;
                for chunk in results.chunks(chunk_size) {
                    scope.spawn(move |_| {
                        let mut idx = start_idx;
                        for e in chunk.iter() {
                            let mut el = *e;
                            el.mul_assign(&mul_by);

                            for map in maps_ref.iter() {
                                if let Some(val) = map.get(&el) {
                                    let value = idx as u32;
                                    let this_rotated_value = rotate_left_32(val.0, val.1 as usize);
                                    let other_rotated_value = rotate_left_32(value, shift_2);
                                    if this_rotated_value != other_rotated_value {
                                        panic!("Same encoding of {} for value {:#b}, shift {} and value {:#b}, shift {}", el, val.0, val.1, value, shift_2);
                                    }
                                }
                            }
                            idx += 1;
                        }
                    });

                    start_idx += chunk_size;
                }
            });

            println!("Verification taken {:?}", start.elapsed());

            println!("Finished checking shift {} vs shift {}", shift_1, shift_2);
        }
    }
}