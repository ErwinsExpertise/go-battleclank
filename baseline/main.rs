use actix_web::{
    App, HttpServer, post,
    web::{Data, Json},
};
use aspirin::{Board, Details, GameOver, Move, MoveAction, Start};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    sync::OnceLock,
};
use tokio::sync::RwLock;

use crate::direction::TurnLogic;

mod direction;
mod scoring;
mod shouts;
mod size;

static GAMES: OnceLock<RwLock<BTreeMap<String, GameState>>> = OnceLock::new();

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();
    let snake_details = Details {
        apiversion: "1".to_string(),
        author: Some("wiggels".to_string()),
        color: Some("#FFF200".to_string()),
        head: Some("sand-worm".to_string()),
        tail: Some("replit-notmark".to_string()),
        version: None,
    };
    let temp = PathBuf::from("/tmp/snakes");
    std::fs::create_dir_all(&temp).ok();
    let temp = Data::new(temp);
    let snake_details = Data::new(snake_details);
    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("BIND_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8080);
    HttpServer::new(move || {
        App::new()
            .app_data(snake_details.clone())
            .app_data(temp.clone())
            .service(details)
            .service(handle_start)
            .service(handle_move)
            .service(handle_game_over)
    })
    .bind((addr, port))?
    .run()
    .await?;
    Ok(())
}

#[actix_web::get("/")]
async fn details(details: Data<Details>) -> Result<Json<Details>, Error> {
    Ok(Json(details.get_ref().clone()))
}

#[actix_web::post("/start")]
async fn handle_start(start: Json<Start>) -> Result<Json<HashMap<String, String>>, Error> {
    let Start {
        game,
        turn: _,
        board,
        you: _,
    } = start.into_inner();
    let states = get_states();

    states.write().await.insert(
        game.id,
        GameState {
            turns: vec![GameStateTurn { logic: None, board }],
        },
    );
    Ok(Json(HashMap::new()))
}

#[post("/move")]
async fn handle_move(event: Json<Move>) -> Result<Json<MoveAction>, Error> {
    log::info!("-----handle move-----");
    let Move {
        game,
        turn: _,
        board,
        you,
    } = event.into_inner();
    let states = get_states();
    let (longest, _) = size::longest(&board.snakes);
    let turn = direction::calculate_action(&game, &you, &board, longest);
    let action = turn.action;
    states
        .write()
        .await
        .get_mut(&game.id)
        .ok_or_else(|| Error::GameNotFound(game.id.clone()))?
        .turns
        .push(GameStateTurn {
            logic: Some(turn),
            board,
        });
    // 30% chance to shout something
    let shout = if rand::random::<u8>() < 255 / 3 {
        let index = rand::random::<u32>() as usize % shouts::SHOUTS.len();
        let shout = shouts::SHOUTS.get(index).map(|s| String::from(*s));
        log::info!(
            "Shout: {}",
            shout.clone().unwrap_or("No shout!".to_string())
        );
        shout
    } else {
        log::info!("Not shouting...");
        None
    };
    log::info!("--------------------");
    Ok(Json(MoveAction { action, shout }))
}

#[post("end")]
async fn handle_game_over(
    end: Json<GameOver>,
    temp_dir: Data<PathBuf>,
) -> Result<Json<HashMap<String, String>>, Error> {
    let GameOver {
        game,
        turn: _,
        board: _,
        you,
    } = end.into_inner();
    if let Some(state) = get_states().write().await.remove(&game.id)
        && let Ok(json) = serde_json::to_vec_pretty(&state)
    {
        let path = temp_dir.join(format!(
            "{}-{}-{}-{}.json",
            game.ruleset.name, game.id, you.name, you.id
        ));
        std::fs::write(&path, &json).ok();
    }
    Ok(Json(HashMap::new()))
}

fn get_states() -> &'static RwLock<BTreeMap<String, GameState>> {
    GAMES.get_or_init(|| RwLock::new(BTreeMap::new()))
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("Game not found with id: `{0}`")]
    GameNotFound(String),
}

impl actix_web::ResponseError for Error {
    fn status_code(&self) -> actix_web::http::StatusCode {
        match self {
            Self::GameNotFound(_) => actix_web::http::StatusCode::BAD_REQUEST,
        }
    }
    fn error_response(&self) -> actix_web::HttpResponse<actix_web::body::BoxBody> {
        let status = self.status_code();
        actix_web::HttpResponseBuilder::new(status).json(match self {
            Self::GameNotFound(game_id) => serde_json::json!({
                "message": "game not found",
                "id": game_id
            }),
        })
    }
}

fn setup_logging() {
    if std::env::var("RUST_LOG").is_err() {
        env_logger::builder()
            .filter(None, log::LevelFilter::Info)
            .init();
        return;
    }
    env_logger::init();
}

#[derive(Debug, Serialize, Default, Clone)]
struct GameState {
    turns: Vec<GameStateTurn>,
}

#[derive(Debug, Serialize, Clone)]
struct GameStateTurn {
    board: Board,
    logic: Option<TurnLogic>,
}
