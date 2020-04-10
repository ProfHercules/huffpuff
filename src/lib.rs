extern crate crossbeam_utils;

pub mod huffpuff {
  use branchrs::Node;
  use crossbeam_utils::thread;
  use std::collections::{BinaryHeap, HashMap};
  use std::sync::mpsc;
  use std::usize;

  fn get_max_slices(vec: &Vec<u8>, mut lim: usize) -> usize {
    while vec.len() % lim != 0 && lim > 1 {
      lim -= 1;
    }
    lim
  }

  pub fn byte_freq_map(byte_sequence: &Vec<u8>, max_threads: usize) -> HashMap<u8, usize> {
    let n_threads = get_max_slices(byte_sequence, max_threads);
    let slice_size = byte_sequence.len() / n_threads;

    let slices = byte_sequence.chunks(slice_size);
    let (tx, rx) = mpsc::channel();

    thread::scope(|s| {
      for slice in slices {
        let tx_clone = mpsc::Sender::clone(&tx);
        s.spawn(move |_| {
          let mut hashmap = HashMap::with_capacity(slice.len());
          for byte in slice.iter() {
            let freq = hashmap.entry(*byte).or_insert(0);
            *freq += 1;
          }
          tx_clone.send(hashmap).unwrap();
        });
      }
    })
    .unwrap();

    let mut results: HashMap<u8, usize> = HashMap::with_capacity(byte_sequence.len());

    for _ in 0..n_threads {
      let hashmap = rx.recv().unwrap();
      for (key, value) in hashmap.iter() {
        if results.contains_key(key) {
          results.entry(*key).and_modify(|v| *v += *value);
        } else {
          results.insert(*key, *value);
        }
      }
    }

    results
  }

  pub fn get_byte_queue(input: &HashMap<u8, usize>) -> BinaryHeap<Node<usize, u8>> {
    let mut heap: BinaryHeap<Node<usize, u8>> = BinaryHeap::new();

    for (key, value) in input {
      heap.push(Node::new(*value, Some(*key)));
    }

    heap
  }

  pub fn get_byte_tree(mut byte_queue: BinaryHeap<Node<usize, u8>>) -> Node<usize, u8> {
    while byte_queue.len() > 1 {
      let node1 = byte_queue.pop().unwrap();
      let node2 = byte_queue.pop().unwrap();

      let mut parent = Node::new(0, None);
      parent.link_left(node1);
      parent.link_right(node2);

      byte_queue.push(parent);
    }

    let root_node = byte_queue.pop().unwrap();
    root_node
  }

  pub fn get_huffman_map(tree_root_node: &Node<usize, u8>) -> HashMap<u8, String> {
    let mut huffman_map: HashMap<u8, String> = HashMap::new();

    tree_root_node.walk(String::from(""), &mut huffman_map);

    huffman_map
  }

  pub fn get_encoded_vec(input: &Vec<u8>, huffman_map: &HashMap<u8, String>) -> Vec<u8> {
    let mut result = Vec::new();

    let mut buffer = String::with_capacity(8);
    for byte in input.iter() {
      let encoded_byte_string = huffman_map.get(&byte).expect("Map error, key not found!");

      for ch in encoded_byte_string.chars() {
        buffer.push(ch);
        if buffer.len() == 8 {
          let buffer_byte = u8::from_str_radix(&buffer, 2).unwrap();
          result.push(buffer_byte);
          buffer.clear();
        }
      }
    }

    assert!(buffer.len() < 8);
    let padding_len = 8 - buffer.len() as u8;
    if buffer.len() > 0 {
      let byte = u8::from_str_radix(&buffer, 2).unwrap();
      result.push(byte);
    }
    result.push(padding_len);

    result
  }

  fn decode_huffman_map(huffman_map: &HashMap<u8, String>) -> HashMap<&str, u8> {
    let mut result = HashMap::new();

    for (key, value) in huffman_map {
      result.insert(&value[..], *key);
    }

    result
  }

  pub fn get_decoded_vec(encoded: &Vec<u8>, huffman_map: &HashMap<u8, String>) -> Vec<u8> {
    assert!(!huffman_map.is_empty());
    let decoded_map = decode_huffman_map(huffman_map);
    let longest_bin_sequence = huffman_map.values().max().unwrap().len();

    let mut encoded_iter = encoded.iter();

    let mut result = Vec::with_capacity(encoded.len());

    let mut buffered_key = String::with_capacity(longest_bin_sequence);

    while let Some(encoded_byte) = encoded_iter.next() {
      let mut binary_string = format!("{:08b}", encoded_byte);
      if encoded_iter.len() == 1 {
        let padding_len = encoded_iter.next().unwrap();
        if *padding_len < 8 {
          binary_string = binary_string.split_off(*padding_len as usize);
        }
      }

      for bit in binary_string.chars() {
        buffered_key.push(bit);
        if decoded_map.contains_key(buffered_key.as_str()) {
          let val = decoded_map.get(buffered_key.as_str()).unwrap();
          result.push(*val);
          buffered_key.clear();
        }
      }
    }

    assert_eq!(encoded_iter.len(), 0);
    result
  }

  // pub fn huffman_code_as_bytes(input: &str) -> Vec<u8> {
  //   let mut buf = String::new();
  //   let capacity = (input.len() / 8) + 1;
  //   let mut results = Vec::with_capacity(capacity);

  //   for c in input.chars() {
  //     if buf.len() < 8 {
  //       buf.push(c);
  //     } else {
  //       let val = u8::from_str_radix(&buf, 2).unwrap();
  //       results.push(val);
  //       buf.clear();
  //     }
  //   }
  //   results
  // }

  // pub fn hashmap_as_bytes(input: &HashMap<u8, String>) -> Vec<u8> {
  //   let mut results: Vec<u8> = Vec::new();
  //   let mut buf = String::new();
  //   // [35, 64, 91,
  //   // 65, 58, 49, 50, 51, 59, 66, 58, 52, 53, 54, 59,
  //   // 91, 64, 35] = #@[A:123;B:456]@#
  //   let key_val_sep = 58;
  //   let entry_sep = 59;
  //   // start hashmap as #@[
  //   results.push(35);
  //   results.push(64);
  //   results.push(91);

  //   for (key, val) in input.iter() {
  //     results.push(*key);
  //     while buf.len() < 8 {
  //       for byte in val.bytes() {
  //         // generate individual byte sequences
  //         // and place them into the vec
  //       }
  //     }
  //     results.push(entry_sep);
  //   }

  //   // end hashmap as ]@#
  //   results.push(93);
  //   results.push(64);
  //   results.push(35);
  //   results
  // }

  // #[derive(Debug)]
  // pub struct Symbol {
  //   pub symbol: u8,
  //   pub freq: usize,
  // }
  // impl fmt::Display for Symbol {
  //   fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
  //     write!(f, "'{}' : {}", self.symbol, self.freq,)
  //   }
  // }

  // impl Symbol {
  //   pub fn new(symbol: u8, freq: usize) -> Symbol {
  //     Symbol { symbol, freq }
  //   }
  // }

  // impl Eq for Symbol {}
  // impl PartialEq for Symbol {
  //   fn eq(&self, other: &Self) -> bool {
  //     self.freq == other.freq
  //   }
  // }

  // impl Ord for Symbol {
  //   fn cmp(&self, other: &Symbol) -> Ordering {
  //     (&other.freq).cmp(&self.freq)
  //   }
  // }

  // // `PartialOrd` needs to be implemented as well.
  // impl PartialOrd for Symbol {
  //   fn partial_cmp(&self, other: &Symbol) -> Option<Ordering> {
  //     Some(self.cmp(other))
  //   }
  // }

  #[cfg(test)]
  mod tests {
    use super::*;

    #[test]
    fn test_get_max_slices() {
      for lim_val in 1..128 as usize {
        let ch = 65 + (lim_val % 26) as u8;
        let vec: Vec<u8> = vec![ch; lim_val];
        let max_slices = get_max_slices(&vec, 16);

        assert!(max_slices >= 1);
      }
    }

    fn freq_sum(map: &HashMap<u8, usize>) -> usize {
      let mut sum: usize = 0;
      for (_, value) in map {
        sum += value;
      }
      sum
    }

    fn known_string_vec(str_len: usize) -> (Vec<u8>, usize, usize) {
      let string = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

      let vec: Vec<u8> = string.repeat(str_len).into_bytes();
      let key_count = string.len();
      let frequency_sum = string.len() * str_len;

      (vec, key_count, frequency_sum)
    }

    #[test]
    fn test_byte_freq_map() {
      let (byte_vec, key_count, frequency_sum) = known_string_vec(10);

      let freq_map = byte_freq_map(&byte_vec, 12);

      assert_eq!(freq_map.keys().len(), key_count);
      assert_eq!(freq_sum(&freq_map), frequency_sum);
    }

    #[test]
    fn test_byte_queue() {
      let (byte_vec, key_count, _) = known_string_vec(10);

      let freq_map = byte_freq_map(&byte_vec, 12);
      let byte_queue = get_byte_queue(&freq_map);

      assert_eq!(byte_queue.len(), key_count);
    }

    #[test]
    fn test_byte_tree() {
      let (byte_vec, key_count, _) = known_string_vec(10);

      let freq_map = byte_freq_map(&byte_vec, 12);
      let byte_queue = get_byte_queue(&freq_map);
      let byte_tree_root = get_byte_tree(byte_queue);

      let mut map = HashMap::new();
      byte_tree_root.walk(String::from(""), &mut map);

      assert_eq!(map.keys().len(), key_count);
    }

    #[test]
    fn test_huffman_map() {
      let (byte_vec, key_count, _) = known_string_vec(10);

      let freq_map = byte_freq_map(&byte_vec, 12);
      let byte_queue = get_byte_queue(&freq_map);
      let byte_tree_root = get_byte_tree(byte_queue);
      let huffman_map = get_huffman_map(&byte_tree_root);

      let mut manual_map = HashMap::new();
      byte_tree_root.walk(String::from(""), &mut manual_map);

      assert_eq!(manual_map.keys().len(), key_count);
      assert_eq!(huffman_map, manual_map);
    }

    #[test]
    fn test_get_encoded_vec() {
      let (byte_vec, _, _) = known_string_vec(10);

      let freq_map = byte_freq_map(&byte_vec, 12);
      let byte_queue = get_byte_queue(&freq_map);
      let byte_tree_root = get_byte_tree(byte_queue);
      let huffman_map = get_huffman_map(&byte_tree_root);
      let encoded_vec = get_encoded_vec(&byte_vec, &huffman_map);
    }
    use std::fs::File;
    use std::io::Read;
    fn read_to_vec(filename: &str) -> std::io::Result<Vec<u8>> {
      let mut file = File::open(filename)?;

      let mut data = Vec::new();
      file.read_to_end(&mut data)?;

      return Ok(data);
    }

    #[test]
    fn test_get_decoded_vec() {
      // let (byte_vec, _, _) = known_string_vec(10);

      let byte_vec = vec![
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
      ];

      let freq_map = byte_freq_map(&byte_vec, 12);
      let byte_queue = get_byte_queue(&freq_map);
      let byte_tree_root = get_byte_tree(byte_queue);
      let huffman_map = get_huffman_map(&byte_tree_root);
      let encoded_vec = get_encoded_vec(&byte_vec, &huffman_map);
      let decoded_vec = get_decoded_vec(&encoded_vec, &huffman_map);

      println!("{:?}", encoded_vec);

      assert_eq!(
        String::from_utf8(byte_vec).unwrap(),
        String::from_utf8(decoded_vec).unwrap()
      );
    }

    fn test_file(filename: &str) -> (usize, usize) {
      let byte_vec = read_to_vec(filename).unwrap();

      let freq_map = byte_freq_map(&byte_vec, 12);
      let byte_queue = get_byte_queue(&freq_map);
      let byte_tree_root = get_byte_tree(byte_queue);
      let huffman_map = get_huffman_map(&byte_tree_root);
      let encoded_vec = get_encoded_vec(&byte_vec, &huffman_map);
      let decoded_vec = get_decoded_vec(&encoded_vec, &huffman_map);

      assert_eq!(byte_vec, decoded_vec);
      (byte_vec.len(), encoded_vec.len())
    }

    #[test]
    #[ignore]
    fn test_get_decoded_vec_external_file() {
      let filenames = vec![
        "plrabn12.txt",
        "plrabn12.zip",
        "input.txt",
        "input2.txt",
        "pdf_test.pdf",
      ];

      for filename in filenames {
        let (input_bytes, output_bytes) = test_file(filename);

        let input_size_mb: f64 = input_bytes as f64 / 1024f64.powf(2.0);
        let output_size_mb: f64 = output_bytes as f64 / 1024f64.powf(2.0);

        println!(
          "\n{}:\nCompressed: {:.2} MB -> {:.2} MB ({} bytes -> {} bytes)",
          filename, input_size_mb, output_size_mb, input_bytes, output_bytes
        );
        println!(
          "     Ratio: {:.2}",
          input_bytes as f64 / output_bytes as f64,
        );
      }
    }

    fn test_file_split(filename: &str) -> (usize, usize) {
      let byte_vec = read_to_vec(filename).unwrap();
      let slices = byte_vec.chunks(64);

      let mut output_bytes = 0;

      for slice in slices {
        let slice_vec = slice.to_vec();
        let freq_map = byte_freq_map(&slice_vec, 12);
        let byte_queue = get_byte_queue(&freq_map);
        let byte_tree_root = get_byte_tree(byte_queue);
        let huffman_map = get_huffman_map(&byte_tree_root);
        let encoded_vec = get_encoded_vec(&slice_vec, &huffman_map);
        let decoded_vec = get_decoded_vec(&encoded_vec, &huffman_map);

        output_bytes += encoded_vec.len();
        assert_eq!(slice_vec, decoded_vec);
      }
      (byte_vec.len(), output_bytes)
    }

    #[test]
    #[ignore]
    fn test_get_decoded_vec_external_file_split() {
      let filenames = vec![
        "plrabn12.txt",
        "plrabn12.zip",
        "input.txt",
        "input2.txt",
        "pdf_test.pdf",
      ];

      for filename in filenames {
        let (input_bytes, output_bytes) = test_file_split(filename);

        let input_size_mb: f64 = input_bytes as f64 / 1024f64.powf(2.0);
        let output_size_mb: f64 = output_bytes as f64 / 1024f64.powf(2.0);

        println!(
          "\nSplit {}:\nCompressed: {:.2} MB -> {:.2} MB ({} bytes -> {} bytes)",
          filename, input_size_mb, output_size_mb, input_bytes, output_bytes
        );
        println!(
          "     Ratio: {:.2}",
          input_bytes as f64 / output_bytes as f64,
        );
      }
    }

    // Benchmarking
    #[test]
    fn test_rand_byte_vec() {
      for i in 0..=256 {
        rand_byte_vec(i);
      }
      for i in 0..=16 {
        rand_byte_vec(2usize.pow(i));
      }
    }

    extern crate rand;
    use rand::Rng;
    use std::time::Instant;
    fn rand_byte_vec(len: usize) -> Vec<u8> {
      let mut rng = rand::thread_rng();
      let mut result: Vec<u8> = Vec::with_capacity(len);

      if len <= 255 {
        for i in 0..len {
          result.push(i as u8);
        }
      } else {
        for i in 0..=255 {
          result.push(i);
        }

        for _ in 256..len {
          result.push(rng.gen());
        }
      }

      assert_eq!(len, result.len());

      result
    }
    use std::collections::BTreeMap;
    #[test]
    #[ignore]
    fn bench_byte_freq_map() {
      let byte_vec_len: usize = 2usize.pow(20);

      println!(
        "Simulating file with size {} bytes (~{} MB)",
        byte_vec_len,
        byte_vec_len / 1024 / 1024
      );
      let vec = rand_byte_vec(byte_vec_len);

      let mut elapsed_vec: BTreeMap<_, std::time::Duration> = BTreeMap::new();

      println!("Starting test...");
      for i in 1..=12 {
        let now = Instant::now();
        byte_freq_map(&vec, i);
        let elapsed = now.elapsed();
        elapsed_vec.insert(i, elapsed);
        println!("Elapsed {}: {:#?}", i, elapsed);
      }

      println!("Elapsed: {:#?}", elapsed_vec);
    }
  }
}
