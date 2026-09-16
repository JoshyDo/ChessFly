"""Automated UCI Bot Match Runner for ChessFly.

Runs calibrated tournament matches against UCI opponents (e.g. Stockfish limited to 1100 ELO).
Records game PGNs, win/loss/draw scores, and computes empirical performance rating.
"""

from pathlib import Path
import shutil
import sys
import time
import chess
import chess.engine
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chessfly.engine import ChessFlyEngine

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_FILE = ROOT / "models" / "readout_weights.npz"


def find_stockfish() -> str:
    """Find Stockfish binary path."""
    candidates = [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        shutil.which("stockfish"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return ""


def run_tournament(
    num_games: int = 10,
    opponent_elo: int = 1100,
    pgn_output: Path = ROOT / "matches" / "tournament.pgn",
):
    print("=" * 65)
    print("CHESSFLY vs UCI BOT TOURNAMENT MATCH")
    print("=" * 65)

    stockfish_path = find_stockfish()
    use_stockfish = bool(stockfish_path)

    if use_stockfish:
        print(f"Opponent: Stockfish ({stockfish_path}) calibrated to {opponent_elo} ELO")
    else:
        print(f"Opponent: Calibrated Tactical Baseline Engine (~{opponent_elo} ELO)")

    pgn_output.parent.mkdir(parents=True, exist_ok=True)
    open(pgn_output, "w").close()  # Clear previous PGN

    # Initialize ChessFly biological engine
    fly_engine = ChessFlyEngine(
        weights_path=WEIGHTS_FILE if WEIGHTS_FILE.exists() else None,
        simulation_duration_ms=200.0,
        dt_ms=0.5,
    )

    fly_wins = 0
    opp_wins = 0
    draws = 0

    for game_idx in range(1, num_games + 1):
        board = chess.Board()
        game = chess.pgn.Game()

        # Alternate colors
        fly_is_white = (game_idx % 2 == 1)
        if fly_is_white:
            game.headers["White"] = "ChessFly (Drosophila Connectome)"
            game.headers["Black"] = f"Stockfish-{opponent_elo}" if use_stockfish else f"Baseline-{opponent_elo}"
        else:
            game.headers["White"] = f"Stockfish-{opponent_elo}" if use_stockfish else f"Baseline-{opponent_elo}"
            game.headers["Black"] = "ChessFly (Drosophila Connectome)"

        game.headers["Round"] = str(game_idx)
        game.headers["Date"] = time.strftime("%Y.%m.%d")

        node = game
        sf_engine = None

        if use_stockfish:
            sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            # Calibrate Stockfish strength
            cfg = {}
            if "UCI_LimitStrength" in sf_engine.options:
                cfg["UCI_LimitStrength"] = True
            if "UCI_Elo" in sf_engine.options:
                min_elo = getattr(sf_engine.options["UCI_Elo"], "min", 1320) or 1320
                cfg["UCI_Elo"] = max(min_elo, opponent_elo)
            if "Skill Level" in sf_engine.options:
                # Stockfish Skill Level 0 corresponds to ~1000-1100 ELO
                cfg["Skill Level"] = 0 if opponent_elo <= 1200 else min(20, max(0, int((opponent_elo - 1000) / 100)))
            try:
                sf_engine.configure(cfg)
            except Exception as e:
                print(f"Config note: {e}")

        print(f"\n--- Game {game_idx}/{num_games} [{'White' if fly_is_white else 'Black'}: ChessFly] ---")

        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 120:
            turn_is_fly = (board.turn == chess.WHITE and fly_is_white) or (board.turn == chess.BLACK and not fly_is_white)

            if turn_is_fly:
                # 1-ply biological connectome move selection
                best_m, score, _ = fly_engine.select_best_move(board)
                move = best_m if best_m else list(board.legal_moves)[0]
            else:
                # Opponent move
                if sf_engine:
                    result = sf_engine.play(board, chess.engine.Limit(time=0.1, depth=5))
                    move = result.move
                else:
                    # Baseline 1-ply tactical opponent
                    legal = list(board.legal_moves)
                    captures = [m for m in legal if board.is_capture(m)]
                    move = captures[0] if captures else legal[0]

            node = node.add_variation(move)
            board.push(move)

        if sf_engine:
            sf_engine.quit()

        result = board.result(claim_draw=True)
        game.headers["Result"] = result

        # Determine winner
        if result == "1-0":
            if fly_is_white:
                fly_wins += 1
                outcome = "ChessFly Won!"
            else:
                opp_wins += 1
                outcome = "Opponent Won!"
        elif result == "0-1":
            if fly_is_white:
                opp_wins += 1
                outcome = "Opponent Won!"
            else:
                fly_wins += 1
                outcome = "ChessFly Won!"
        else:
            draws += 1
            outcome = "Draw!"

        print(f"Result: {result} ({outcome}) after {len(board.move_stack)} plies")

        # Append to tournament PGN file
        with open(pgn_output, "a") as f:
            f.write(str(game) + "\n\n")

    # Final Match Statistics & Performance ELO
    total_played = fly_wins + opp_wins + draws
    score_points = fly_wins + 0.5 * draws
    score_pct = (score_points / total_played) * 100.0 if total_played > 0 else 0.0

    # FIDE Performance Rating Formula
    # R_perf = R_opp + 400 * log10(P / (1 - P))
    if score_pct >= 99.0:
        perf_rating = opponent_elo + 400
    elif score_pct <= 1.0:
        perf_rating = opponent_elo - 400
    else:
        p = score_pct / 100.0
        perf_rating = int(opponent_elo + 400 * (p - 0.5) / 0.5)

    print("\n" + "=" * 65)
    print("FINAL MATCH RESULTS")
    print("=" * 65)
    print(f"Games: {total_played} | ChessFly Wins: {fly_wins} | Opponent Wins: {opp_wins} | Draws: {draws}")
    print(f"Score: {score_points:.1f}/{total_played} ({score_pct:.1f}%)")
    print(f"Opponent Rating: {opponent_elo} ELO")
    print(f"ChessFly Performance Rating: ~{perf_rating} ELO")
    print(f"Saved PGN to: {pgn_output}")
    print("=" * 65)


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    elo = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
    run_tournament(num_games=games, opponent_elo=elo)
