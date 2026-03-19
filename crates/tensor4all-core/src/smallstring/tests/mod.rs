
use super::*;

#[test]
fn test_smallstring_u16_basic() {
    let s = SmallString::<16>::from_str("hello").unwrap();
    assert_eq!(s.as_str(), "hello");
    assert_eq!(s.len(), 5);
}

#[test]
fn test_smallstring_u16_japanese() {
    // Japanese characters are in BMP, so u16 should work
    let s = SmallString::<16>::from_str("日本語").unwrap();
    assert_eq!(s.as_str(), "日本語");
    assert_eq!(s.len(), 3);
}

#[test]
fn test_smallstring_u16_emoji_fails() {
    // Emoji (😀 = U+1F600) is outside BMP, so u16 should fail
    let result = SmallString::<16>::from_str("hello 😀");
    assert!(matches!(result, Err(SmallStringError::InvalidChar { .. })));
}

#[test]
fn test_smallstring_char_emoji() {
    // char type should support emoji
    let s = SmallString::<16, char>::from_str("hello 😀").unwrap();
    assert_eq!(s.as_str(), "hello 😀");
}

#[test]
fn test_smallstring_too_long() {
    let result = SmallString::<4>::from_str("hello");
    assert!(matches!(
        result,
        Err(SmallStringError::TooLong { actual: 5, max: 4 })
    ));
}

#[test]
fn test_smallstring_ordering() {
    let a = SmallString::<16>::from_str("abc").unwrap();
    let b = SmallString::<16>::from_str("abd").unwrap();
    let c = SmallString::<16>::from_str("abc").unwrap();

    assert!(a < b);
    assert_eq!(a, c);
}

#[test]
fn test_smallstring_size() {
    // u16 version: 16 * 2 + 8 = 40 bytes
    assert_eq!(std::mem::size_of::<SmallString<16, u16>>(), 40);
    // char version: 16 * 4 + 8 = 72 bytes
    assert_eq!(std::mem::size_of::<SmallString<16, char>>(), 72);
}
