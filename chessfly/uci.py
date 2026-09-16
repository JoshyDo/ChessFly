"""Universal Chess Interface (UCI) protocol driver for ChessFly biological connectome engine."""

import sys
import chess
from chessfly.engine import ChessFlyEngine


def run_uci():
    """Main UCI protocol loop for standard chess GUIs."""
    engine = ChessFlyEngine(simulation_duration_ms=500.0, dt_ms=0.5)
    board = chess.Board()

    sys.stdout.write("id name ChessFly Biological Connectome\n")
    sys.stdout.write("id author Google Deepmind Advanced Agentic Coding\n")
    sys.stdout.write("uciok\n")
    sys.stdout.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        tokens = line.strip().split()
        if not tokens:
            continue

        cmd = tokens[0]

        if cmd == "isready":
            sys.stdout.write("readyok\n")
            sys.stdout.flush()
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            if "startpos" in tokens:
                board = chess.Board()
                if "moves" in tokens:
                    moves_idx = tokens.index("moves") + 1
                    for m in tokens[moves_idx:]:
                        board.push_uci(m)
            elif "fen" in tokens:
                fen_idx = tokens.index("fen") + 1
                moves_idx = tokens.index("moves") if "moves" in tokens else len(tokens)
                fen_str = " ".join(tokens[fen_idx:moves_idx])
                board = chess.Board(fen_str)
                if "moves" in tokens:
                    for m in tokens[moves_idx + 1:]:
                        board.push_uci(m)
        elif cmd == "go":
            best_move, score, _ = engine.select_best_move(board)
            if best_move:
                sys.stdout.write(f"bestmove {best_move.uci()}\n")
            else:
                sys.stdout.write("bestmove (none)\n")
            sys.stdout.flush()
        elif cmd == "quit":
            break


if __name__ == "__main__":
    run_uci()
