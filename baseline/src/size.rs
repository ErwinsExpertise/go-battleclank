use aspirin::BattleSnake;

/// calculate who is the longest snake in the provided slice
pub fn longest(snakes: &[BattleSnake]) -> (&str, u32) {
    snakes.iter().fold(("", 0), |(longest_name, length), s| {
        if length < s.length {
            return (s.id.as_str(), s.length);
        }
        (longest_name, length)
    })
}
