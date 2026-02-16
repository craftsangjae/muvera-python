/// Append a single bit to a Gray code value.
#[inline]
pub fn append_to_gray_code(gray_code: u32, bit: bool) -> u32 {
    (gray_code << 1) + ((bit as u32) ^ (gray_code & 1))
}

/// Convert a Gray code value to its binary representation.
#[inline]
pub fn gray_code_to_binary(mut num: u32) -> u32 {
    let mut mask = num >> 1;
    while mask != 0 {
        num ^= mask;
        mask >>= 1;
    }
    num
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_code_roundtrip() {
        // Gray code for 0..8 should produce unique values
        let mut results = Vec::new();
        for i in 0u32..8 {
            let bits = [i & 4 != 0, i & 2 != 0, i & 1 != 0];
            let mut gc = 0u32;
            for &b in &bits {
                gc = append_to_gray_code(gc, b);
            }
            results.push(gc);
        }
        // All should be unique
        results.sort();
        results.dedup();
        assert_eq!(results.len(), 8);
    }

    #[test]
    fn test_gray_code_to_binary() {
        assert_eq!(gray_code_to_binary(0), 0);
        assert_eq!(gray_code_to_binary(1), 1);
        assert_eq!(gray_code_to_binary(3), 2);
        assert_eq!(gray_code_to_binary(2), 3);
    }
}
