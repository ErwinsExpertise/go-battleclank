use aspirin::{Action, BattleSnake, Board, Game};
use serde::Serialize;

use crate::scoring::{ScoredMove, ScoringWeights, score_moves, select_best_move};

/// Calculate the current direction of the snake
pub fn snake_direction(snake: &BattleSnake) -> Option<Action> {
    if snake.body.len() < 2 {
        return None;
    }

    let neck = snake.body[1];
    if neck == snake.head {
        return None;
    }
    if neck.x < snake.head.x {
        return Some(Action::Right);
    }
    if neck.x > snake.head.x {
        return Some(Action::Left);
    }
    if neck.y < snake.head.y {
        return Some(Action::Up);
    }
    Some(Action::Down)
}

/// Calculate a good next action to take using the weighted scoring system
pub fn calculate_action(
    game: &Game,
    snake: &BattleSnake,
    board: &Board,
    longest: &str,
) -> TurnLogic {
    log::info!(
        "calculate_action: game={:?}, head={:?}, body={:?}, health={}, longest={longest}",
        game.ruleset.name,
        snake.head,
        snake.body,
        snake.health,
    );
    let current_direction = snake_direction(snake);
    log::info!("current direction: {:?}", current_direction);
    // pick weight strategy based on game
    let weights = match game.ruleset.name.as_str() {
        "royale" => {
            // royale shrinks board -- aggressive food + center control
            // for now -- reg
            ScoringWeights::new()
        }
        "constrictor" => {
            // no food spawns -- pure survival + space control
            // for now -- reg
            ScoringWeights::new()
        }
        _ => {
            // standard mode
            if snake.id == longest {
                // we are biggest -- play aggressive
                ScoringWeights::new().with_combat_aggression(1.0)
            } else {
                // default balanced
                ScoringWeights::new()
            }
        }
    };
    let scored_moves = score_moves(snake, board, current_direction, &weights);
    let best_move = select_best_move(&scored_moves);
    log::info!("selected move: {:?}", best_move);
    TurnLogic {
        weights,
        scored: scored_moves,
        action: best_move,
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct TurnLogic {
    pub weights: ScoringWeights,
    pub scored: Vec<ScoredMove>,
    pub action: Action,
}
