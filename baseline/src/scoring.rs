use aspirin::{Action, BattleSnake, Board, Coord};
use serde::Serialize;
use std::collections::HashMap;

// scoring engine -- evaluates moves based on survival + food + space control
// higher score = better move, -1000 = instant death
// breakdowns track individual component contributions for debugging

#[derive(Debug, Clone, Serialize)]
pub struct ScoringWeights {
    // instant death penalties
    pub wall_collision: i32,
    pub self_collision: i32,
    pub other_snake_collision: i32,

    // head to head combat
    pub win_head_to_head: i32,
    pub lose_head_to_head: i32,

    // food attraction weights -- scale with distance
    pub food_adjacent: i32,
    pub food_near: i32,
    pub food_medium: i32,
    pub food_when_hungry: i32,
    pub food_when_starving: i32,
    pub food_death_trap: i32,

    // space analysis -- trap detection uses ratios not absolutes
    pub open_space: i32,
    pub trap_penalty: i32,
    pub critical_trap_penalty: i32,
    pub severe_trap_penalty: i32,

    // positioning heuristics
    pub center_control: i32,
    pub maintain_direction: i32,
    pub near_wall_penalty: i32,
    pub hazard_penalty: i32,
    pub hazard_when_healthy: i32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        ScoringWeights {
            // collisions
            wall_collision: -1000,
            self_collision: -1000,
            other_snake_collision: -1000,

            // combat scoring
            win_head_to_head: 500,
            lose_head_to_head: -500,

            // food proximity
            food_adjacent: 30,
            food_near: 15,
            food_medium: 5,
            food_when_hungry: 40,
            food_when_starving: 80,
            food_death_trap: -800, // eating here = trapped

            // space thresholds -- balanced to avoid false positives
            // trap detection uses space/body ratios not absolute counts
            open_space: 5,               // reward per accessible square
            trap_penalty: -250,          // 60-80% space ratio -- mild warning
            critical_trap_penalty: -600, // <40% space ratio -- deadly
            severe_trap_penalty: -450,   // 40-60% space ratio -- very bad

            // positioning bonuses
            center_control: 2,
            maintain_direction: 1,
            near_wall_penalty: -5,

            // hazards
            hazard_penalty: -30,
            hazard_when_healthy: -10,
        }
    }
}

impl ScoringWeights {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_combat_aggression(mut self, multiplier: f32) -> Self {
        self.win_head_to_head = (self.win_head_to_head as f32 * multiplier) as i32;
        self
    }
}

// move with total score + component breakdown for debugging
#[derive(Debug, Serialize, Clone)]
pub struct ScoredMove {
    pub action: Action,
    pub total_score: i32,
    pub score_breakdown: HashMap<String, i32>,
}

// evaluate all 4 directions -- return scored moves sorted by quality
pub fn score_moves(
    snake: &BattleSnake,
    board: &Board,
    current_direction: Option<Action>,
    weights: &ScoringWeights,
) -> Vec<ScoredMove> {
    let possible_moves = vec![Action::Up, Action::Down, Action::Left, Action::Right];

    let scored_moves: Vec<ScoredMove> = possible_moves
        .into_iter()
        .map(|action| {
            let mut score_breakdown = HashMap::new();
            let total_score = calculate_move_score(
                snake,
                board,
                action,
                current_direction,
                weights,
                &mut score_breakdown,
            );

            ScoredMove {
                action,
                total_score,
                score_breakdown,
            }
        })
        .collect();

    scored_moves
}

// main scoring logic -- applies all heuristics to a single move
// returns early with -1000 for instant death scenarios
fn calculate_move_score(
    snake: &BattleSnake,
    board: &Board,
    action: Action,
    current_direction: Option<Action>,
    weights: &ScoringWeights,
    breakdown: &mut HashMap<String, i32>,
) -> i32 {
    let mut score = 0;

    // check wall collision first
    let next_pos = match next_coord(snake.head, action, board.width, board.height) {
        Some(pos) => pos,
        None => {
            breakdown.insert("wall_collision".to_string(), weights.wall_collision);
            return weights.wall_collision;
        }
    };

    // self collision -- need to check if tail will actually move
    // tail stays if we just grew or are about to grow
    if snake.body.contains(&next_pos) {
        let already_growing = snake.body.len() >= 2
            && snake.body[snake.body.len() - 1] == snake.body[snake.body.len() - 2];
        let will_grow = board.food.contains(&next_pos);
        let tail_will_move = !already_growing && !will_grow;

        if snake.body.last() != Some(&next_pos) || !tail_will_move {
            breakdown.insert("self_collision".to_string(), weights.self_collision);
            return weights.self_collision;
        }
    }

    // check collisions with other snakes -- similar tail logic applies
    for other_snake in &board.snakes {
        if other_snake.id == snake.id {
            continue;
        }

        if other_snake.body.contains(&next_pos) {
            let tail_will_stay = other_snake.body.len() >= 2
                && other_snake.body[other_snake.body.len() - 1]
                    == other_snake.body[other_snake.body.len() - 2];

            if other_snake.body.last() != Some(&next_pos) || tail_will_stay {
                breakdown.insert(
                    "other_snake_collision".to_string(),
                    weights.other_snake_collision,
                );
                return weights.other_snake_collision;
            }
        }

        // head to head collision detection
        // if we are next to their head after move -- compare lengths
        let dist = distance(&next_pos, &other_snake.head);
        if dist == 1 {
            if snake.length > other_snake.length {
                let points = weights.win_head_to_head;
                breakdown.insert("win_head_to_head".to_string(), points);
                score += points;
            } else {
                let points = weights.lose_head_to_head;
                breakdown.insert("lose_head_to_head".to_string(), points);
                score += points;
            }
        }

        // aggressive pursuit -- move toward enemy heads if we outsize them
        // scale inversely with distance -- closer = more reward
        if snake.length > other_snake.length && dist <= 5 {
            let pursuit_bonus = match dist {
                2 => 100, // almost in range -- very good
                3 => 50,  // closing in
                4 => 25,  // still relevant
                5 => 10,  // on radar
                _ => 0,
            };

            if pursuit_bonus > 0 {
                breakdown.insert(format!("hunt_snake_{}", other_snake.id), pursuit_bonus);
                score += pursuit_bonus;
            }
        }
    }

    // flood fill to count accessible squares
    let open_squares = count_accessible_squares(&next_pos, snake, board, false);
    let space_score = (open_squares as i32) * weights.open_space;
    breakdown.insert("open_space".to_string(), space_score);
    score += space_score;

    // trap detection using space/body ratios
    // key insight -- tail moves so dont need 100% of body length in space
    // thresholds tuned from playtesting to avoid false positives
    let body_length = snake.length as usize;
    let space_ratio = open_squares as f32 / body_length as f32;

    // <40% space = critical danger -- less than half what you need
    if space_ratio < 0.4 {
        breakdown.insert("critical_trap".to_string(), weights.critical_trap_penalty);
        score += weights.critical_trap_penalty;
        log::error!(
            "CRITICAL TRAP {:?}: {} spaces / {} length = {:.0}%",
            next_pos,
            open_squares,
            body_length,
            space_ratio * 100.0
        );
    }
    // 40-60% space = severe danger -- getting boxed in
    else if space_ratio < 0.6 {
        breakdown.insert("severe_trap".to_string(), weights.severe_trap_penalty);
        score += weights.severe_trap_penalty;
        log::warn!(
            "SEVERE TRAP {:?}: {} spaces / {} length = {:.0}%",
            next_pos,
            open_squares,
            body_length,
            space_ratio * 100.0
        );
    }
    // 60-80% space = moderate concern -- worth noting but not critical
    else if space_ratio < 0.8 {
        breakdown.insert("trap_penalty".to_string(), weights.trap_penalty);
        score += weights.trap_penalty;
        log::info!(
            "Moderate trap {:?}: {} spaces / {} length = {:.0}%",
            next_pos,
            open_squares,
            body_length,
            space_ratio * 100.0
        );
    }
    // 80%+ space = good -- tail is moving so this is safe
    else {
        log::debug!(
            "Good space {:?}: {} spaces / {} length = {:.0}%",
            next_pos,
            open_squares,
            body_length,
            space_ratio * 100.0
        );
    }

    // lookahead -- penalize moves that lead to progressively tighter spaces
    // simulates one move ahead and checks if space ratio gets worse
    if open_squares > 0 {
        let current_ratio = open_squares as f32 / body_length as f32;

        // simulate next moves from this position
        let mut worst_next_ratio = current_ratio;
        for next_action in &[Action::Up, Action::Down, Action::Left, Action::Right] {
            if let Some(future_pos) = next_coord(next_pos, *next_action, board.width, board.height)
            {
                let future_spaces = count_accessible_squares(&future_pos, snake, board, false);
                let future_ratio = future_spaces as f32 / body_length as f32;
                worst_next_ratio = worst_next_ratio.min(future_ratio);
            }
        }

        // if all future moves lead to worse space, heavily penalize
        if worst_next_ratio < current_ratio * 0.8 {
            let dead_end_penalty = -200;
            breakdown.insert("dead_end_ahead".to_string(), dead_end_penalty);
            score += dead_end_penalty;
            log::warn!(
                "dead end ahead at {:?}: current ratio {:.0}%, worst next {:.0}%",
                next_pos,
                current_ratio * 100.0,
                worst_next_ratio * 100.0
            );
        }
    }

    // food death trap detection -- different threshold since tail wont move after eating
    // need 70% of body length since tail stays put
    if board.food.contains(&next_pos) {
        // check if we just ate -- if so we are already growing and this is fine
        let already_growing = snake.body.len() >= 2
            && snake.body[snake.body.len() - 1] == snake.body[snake.body.len() - 2];

        if !already_growing {
            let spaces_after_eating = count_accessible_squares(&next_pos, snake, board, true);
            let min_required = (body_length as f32 * 0.7) as usize; // 70% threshold

            if spaces_after_eating < min_required {
                breakdown.insert("food_death_trap".to_string(), weights.food_death_trap);
                score += weights.food_death_trap;
                log::error!(
                    "FOOD TRAP {:?}: {} spaces after eating -- need {}+ -- 70% of {}",
                    next_pos,
                    spaces_after_eating,
                    min_required,
                    body_length
                );
            } else {
                let food_score = score_food_proximity(snake, &next_pos, board, weights);
                if food_score != 0 {
                    breakdown.insert("food_proximity".to_string(), food_score);
                    score += food_score;
                }
            }
        } else {
            let food_score = score_food_proximity(snake, &next_pos, board, weights);
            if food_score != 0 {
                breakdown.insert("food_proximity".to_string(), food_score);
                score += food_score;
            }
        }
    } else {
        let food_score = score_food_proximity(snake, &next_pos, board, weights);
        if food_score != 0 {
            breakdown.insert("food_proximity".to_string(), food_score);
            score += food_score;
        }
    }

    // health urgency -- boost food value when low hp
    if snake.health < 30 {
        let urgency = if snake.health < 15 {
            weights.food_when_starving
        } else {
            weights.food_when_hungry
        };

        if board.food.iter().any(|f| distance(&next_pos, f) < 5) {
            breakdown.insert("health_urgency".to_string(), urgency);
            score += urgency;
        }
    }

    // center control -- prefer middle of board over edges
    // score scales from 0 at edges to max at center
    let board_center_x = board.width as f32 / 2.0;
    let board_center_y = board.height as f32 / 2.0;
    let center_distance = ((next_pos.x as f32 - board_center_x).powi(2)
        + (next_pos.y as f32 - board_center_y).powi(2))
    .sqrt();
    let max_distance = (board_center_x.powi(2) + board_center_y.powi(2)).sqrt();
    let center_score =
        ((1.0 - center_distance / max_distance) * weights.center_control as f32).round() as i32;
    if center_score != 0 {
        breakdown.insert("center_control".to_string(), center_score);
        score += center_score;
    }

    // slight bonus for continuing in same direction -- reduces thrashing
    if let Some(prev_dir) = current_direction
        && action == prev_dir
    {
        breakdown.insert("maintain_direction".to_string(), weights.maintain_direction);
        score += weights.maintain_direction;
    }

    // penalize being near walls -- reduces options
    let dist_to_wall = wall_distance(&next_pos, board);
    if dist_to_wall <= 1 {
        breakdown.insert("near_wall_penalty".to_string(), weights.near_wall_penalty);
        score += weights.near_wall_penalty;
    }

    // hazard damage -- less penalty if healthy enough to tank it
    if !board.hazards.is_empty() && board.hazards.contains(&next_pos) {
        let penalty = if snake.health > 50 {
            weights.hazard_when_healthy
        } else {
            weights.hazard_penalty
        };
        breakdown.insert("hazard".to_string(), penalty);
        score += penalty;
    }

    score
}

// calculate food attraction score based on distance
// scales with health -- hungrier = higher multiplier
fn score_food_proximity(
    snake: &BattleSnake,
    pos: &Coord,
    board: &Board,
    weights: &ScoringWeights,
) -> i32 {
    let mut best_score = 0;

    for food in &board.food {
        let dist = distance(pos, food);

        let score = match dist {
            0 => weights.food_adjacent * 2,
            1 => weights.food_adjacent,
            2..=3 => weights.food_near,
            4..=5 => weights.food_medium,
            _ => 0,
        };

        let health_multiplier = if snake.health < 20 {
            2.0
        } else if snake.health < 40 {
            1.5
        } else {
            1.0
        };

        let adjusted_score = (score as f32 * health_multiplier) as i32;
        best_score = best_score.max(adjusted_score);
    }

    best_score
}

// bfs flood fill to count reachable squares from start position
// will_grow param determines if we treat tail as blocking or not
fn count_accessible_squares(
    start: &Coord,
    snake: &BattleSnake,
    board: &Board,
    will_grow: bool,
) -> usize {
    use std::collections::VecDeque;

    let mut visited: Vec<Coord> = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(*start);
    visited.push(*start);

    let board_area = (board.width * board.height) as usize;
    let max_search = board_area.min(121); // cap at 11x11 for perf

    let mut count = 0;

    while let Some(pos) = queue.pop_front() {
        count += 1;

        if count >= max_search {
            break;
        }

        for action in &[Action::Up, Action::Down, Action::Left, Action::Right] {
            if let Some(next) = next_coord(pos, *action, board.width, board.height) {
                if visited.contains(&next) {
                    continue;
                }

                let mut is_safe = true;

                // if growing treat full body as blocking
                // otherwise ignore tail since it will move
                if will_grow {
                    if snake.body.contains(&next) {
                        is_safe = false;
                    }
                } else if snake.body.len() > 1 && snake.body[..snake.body.len() - 1].contains(&next)
                {
                    is_safe = false;
                }

                // check other snakes -- ignore their tails too
                for other in &board.snakes {
                    if other.id != snake.id
                        && other.body.len() > 1
                        && other.body[..other.body.len() - 1].contains(&next)
                    {
                        is_safe = false;
                        break;
                    }
                }

                if is_safe {
                    queue.push_back(next);
                    visited.push(next);
                }
            }
        }
    }

    count
}

// manhattan distance to nearest wall
fn wall_distance(pos: &Coord, board: &Board) -> u32 {
    let dist_left = pos.x;
    let dist_right = (board.width as u32) - 1 - pos.x;
    let dist_down = pos.y;
    let dist_up = (board.height as u32) - 1 - pos.y;

    dist_left.min(dist_right).min(dist_down).min(dist_up)
}

// compute next coord from action -- returns none if out of bounds
fn next_coord(start: Coord, direction: Action, max_x: u64, max_y: u64) -> Option<Coord> {
    let Coord { x, y } = start;
    Some(match direction {
        Action::Down => Coord {
            x,
            y: y.checked_sub(1)?,
        },
        Action::Up => {
            let new_y = y + 1;
            if new_y >= max_y as u32 {
                return None;
            }
            Coord { x, y: new_y }
        }
        Action::Left => Coord {
            x: x.checked_sub(1)?,
            y,
        },
        Action::Right => {
            let new_x = x + 1;
            if new_x >= max_x as u32 {
                return None;
            }
            Coord { x: new_x, y }
        }
    })
}

// manhattan distance between two coords
fn distance(a: &Coord, b: &Coord) -> u32 {
    ((a.x as i32 - b.x as i32).abs() + (a.y as i32 - b.y as i32).abs()) as u32
}

// pick highest scoring move -- fallback to up if all moves are fatal
pub fn select_best_move(scored_moves: &[ScoredMove]) -> Action {
    for scored_move in scored_moves {
        log::info!(
            "Move {:?}: score={}, breakdown={:?}",
            scored_move.action,
            scored_move.total_score,
            scored_move.score_breakdown
        );
    }

    let best_move = scored_moves.iter().max_by_key(|m| m.total_score);

    match best_move {
        Some(m) if m.total_score > -1000 => m.action,
        _ => {
            log::error!("all moves fatal -- just die");
            Action::Up
        }
    }
}
