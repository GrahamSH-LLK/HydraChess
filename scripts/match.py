"""Engine-vs-engine matches with an SPRT stop rule, and self-play data generation for tuning.

SPRT (does `new` gain Elo over `base`?):
    uv run match.py sprt NEW BASE [--tc 8+0.08] [--concurrency 8] [--elo0 0 --elo1 5]

Self-play data for `HydraChess tune` (one "FEN | result" line per position, white's result):
    uv run match.py datagen ENGINE OUT.txt [--games 4000] [--nodes 6000] [--concurrency 8]

Openings are random 8-ply walks from the start position, kept only if the engine scores them
as roughly balanced. Every opening is played twice with colors swapped, and the SPRT works on
game pairs (pentanomial), which removes most of the opening bias from the result.
"""

import argparse
import math
import multiprocessing as mp
import random
import sys
import time

import chess
import chess.engine

RANDOM_PLIES = 8
MAX_OPENING_EVAL = 120


def random_opening(engine: chess.engine.SimpleEngine, rng: random.Random) -> chess.Board:
    while True:
        board = chess.Board()
        for _ in range(RANDOM_PLIES):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        info = engine.analyse(board, chess.engine.Limit(depth=6))
        # With a single legal move the engine replies at once, without a score
        if "score" not in info:
            continue
        score = info["score"].white()
        if not score.is_mate() and abs(score.score()) <= MAX_OPENING_EVAL:
            return board


def configure(engine: chess.engine.SimpleEngine, threads: int, hash_mb: int):
    """Sets Threads and Hash where the engine offers them (older versions have fixed values)."""
    options = {"Threads": threads, "Hash": hash_mb}
    engine.configure({k: v for k, v in options.items() if k in engine.options})


def play_game(engines, opening: chess.Board, limit_for, record_positions=False):
    """engines[0] plays white. Returns (result for white in [1, 0.5, 0], positions)."""
    board = opening.copy()
    # python-chess sends ucinewgame on the first search of a new game id
    game = object()
    clocks = None
    positions = []
    win_streak = 0
    draw_streak = 0
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result = 1.0 if outcome.winner is True else 0.0 if outcome.winner is False else 0.5
            break
        engine = engines[0 if board.turn == chess.WHITE else 1]
        limit, clocks = limit_for(board, clocks)
        start = time.perf_counter()
        play = engine.play(board, limit, game=game, info=chess.engine.INFO_SCORE)
        elapsed = time.perf_counter() - start
        if clocks is not None:
            side = 0 if board.turn == chess.WHITE else 1
            clocks[side] = clocks[side] - elapsed + clocks[2]
            if clocks[side] <= 0:
                result = 0.0 if board.turn == chess.WHITE else 1.0
                print(f"time loss for {'white' if side == 0 else 'black'}", file=sys.stderr)
                break

        score = play.info.get("score")
        if record_positions and score is not None and not board.is_check():
            positions.append(board.fen())
        board.push(play.move)

        # Adjudication keeps decided games short
        if score is not None:
            white_score = score.white()
            cp = white_score.score(mate_score=100_000)
            if abs(cp) >= 1000:
                win_streak += 1
            else:
                win_streak = 0
            if board.fullmove_number >= 40 and abs(cp) <= 10:
                draw_streak += 1
            else:
                draw_streak = 0
            if win_streak >= 6:
                result = 1.0 if cp > 0 else 0.0
                break
            if draw_streak >= 10:
                result = 0.5
                break
    return result, positions


def parse_tc(tc: str):
    base, _, increment = tc.partition("+")
    return float(base), float(increment or 0)


def sprt_worker(args, queue, seed):
    rng = random.Random(seed)
    new = chess.engine.SimpleEngine.popen_uci(args.new)
    base = chess.engine.SimpleEngine.popen_uci(args.base)
    configure(new, args.threads, args.hash)
    configure(base, args.threads, args.hash)
    base_time, increment = parse_tc(args.tc)

    def limit_for(board, clocks):
        if clocks is None:
            clocks = [base_time, base_time, increment]
        return (
            chess.engine.Limit(
                white_clock=clocks[0], black_clock=clocks[1], white_inc=increment, black_inc=increment
            ),
            clocks,
        )

    try:
        while True:
            opening = random_opening(base, rng)
            # New engine as white, then as black; scores are from the new engine's side
            first, _ = play_game([new, base], opening, limit_for)
            second, _ = play_game([base, new], opening, limit_for)
            queue.put(first + (1.0 - second))
    finally:
        new.quit()
        base.quit()


def elo_to_score(elo):
    return 1 / (1 + 10 ** (-elo / 400))


def pentanomial_llr(counts, elo0, elo1):
    """GSPRT log-likelihood ratio on game-pair scores (0, 0.5, 1, 1.5, 2), as fishtest does."""
    total = sum(counts)
    if total == 0:
        return 0.0
    # Avoid zero cells
    probs = [(c + 1e-3) / (total + 5e-3) for c in counts]
    values = [0, 0.25, 0.5, 0.75, 1.0]
    mean = sum(p * v for p, v in zip(probs, values))
    variance = sum(p * (v - mean) ** 2 for p, v in zip(probs, values))
    if variance <= 0:
        return 0.0
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return total * (s1 - s0) * (2 * mean - s0 - s1) / (2 * variance)


def elo_estimate(counts):
    total = sum(counts)
    if total == 0:
        return 0.0, 0.0
    values = [0, 0.25, 0.5, 0.75, 1.0]
    mean = sum(c * v for c, v in zip(counts, values)) / total
    variance = sum(c * (v - mean) ** 2 for c, v in zip(counts, values)) / total
    mean = min(max(mean, 1e-6), 1 - 1e-6)
    elo = -400 * math.log10(1 / mean - 1)
    # 95% interval through the delta method
    margin = 1.96 * math.sqrt(variance / total)
    upper = min(mean + margin, 1 - 1e-6)
    elo_upper = -400 * math.log10(1 / upper - 1)
    return elo, elo_upper - elo


def run_sprt(args):
    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)
    queue = mp.Queue()
    workers = [
        mp.Process(target=sprt_worker, args=(args, queue, args.seed + i), daemon=True)
        for i in range(args.concurrency)
    ]
    for worker in workers:
        worker.start()
    counts = [0] * 5
    pairs = 0
    try:
        while args.max_pairs is None or pairs < args.max_pairs:
            pair_score = queue.get()
            counts[int(pair_score * 2 + 0.5)] += 1
            pairs += 1
            llr = pentanomial_llr(counts, args.elo0, args.elo1)
            elo, margin = elo_estimate(counts)
            print(
                f"pairs {pairs} penta {counts} elo {elo:+.1f} ±{margin:.1f} "
                f"llr {llr:.2f} ({lower:.2f}, {upper:.2f})",
                flush=True,
            )
            if args.fixed:
                continue
            if llr >= upper:
                print("H1 accepted: the new engine is stronger")
                break
            if llr <= lower:
                print("H0 accepted: no gain detected")
                break
    finally:
        for worker in workers:
            worker.terminate()


def datagen_worker(args, games, queue, seed):
    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    second = chess.engine.SimpleEngine.popen_uci(args.engine)
    configure(engine, 1, 16)
    configure(second, 1, 16)

    def limit_for(board, clocks):
        # Vary the node count a little so the games aren't all identical
        return chess.engine.Limit(nodes=int(args.nodes * rng.uniform(0.75, 1.25))), None

    try:
        for _ in range(games):
            opening = random_opening(engine, rng)
            result, positions = play_game([engine, second], opening, limit_for, True)
            queue.put((result, positions))
    finally:
        engine.quit()
        second.quit()
        queue.put(None)


def run_datagen(args):
    queue = mp.Queue()
    per_worker = math.ceil(args.games / args.concurrency)
    workers = [
        mp.Process(target=datagen_worker, args=(args, per_worker, queue, args.seed + i), daemon=True)
        for i in range(args.concurrency)
    ]
    for worker in workers:
        worker.start()
    finished = 0
    games = 0
    positions = 0
    with open(args.out, "w") as out:
        while finished < len(workers):
            item = queue.get()
            if item is None:
                finished += 1
                continue
            result, fens = item
            for fen in fens:
                out.write(f"{fen} | {result}\n")
            games += 1
            positions += len(fens)
            if games % 50 == 0:
                print(f"games {games} positions {positions}", flush=True)
    print(f"done: {games} games, {positions} positions in {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    sprt = sub.add_parser("sprt")
    sprt.add_argument("new")
    sprt.add_argument("base")
    sprt.add_argument("--tc", default="8+0.08", help="seconds+increment")
    sprt.add_argument("--concurrency", type=int, default=max(1, mp.cpu_count() // 2))
    sprt.add_argument("--elo0", type=float, default=0)
    sprt.add_argument("--elo1", type=float, default=5)
    sprt.add_argument("--alpha", type=float, default=0.05)
    sprt.add_argument("--beta", type=float, default=0.05)
    sprt.add_argument("--max-pairs", type=int, default=None)
    sprt.add_argument(
        "--fixed", action="store_true", help="play all --max-pairs pairs, ignoring the SPRT bounds"
    )
    sprt.add_argument("--seed", type=int, default=1)
    sprt.add_argument("--threads", type=int, default=1, help="per engine, where supported")
    sprt.add_argument("--hash", type=int, default=16, help="MB per engine, where supported")

    datagen = sub.add_parser("datagen")
    datagen.add_argument("engine")
    datagen.add_argument("out")
    datagen.add_argument("--games", type=int, default=4000)
    datagen.add_argument("--nodes", type=int, default=6000)
    datagen.add_argument("--concurrency", type=int, default=max(1, mp.cpu_count() // 2))
    datagen.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    if args.mode == "sprt":
        run_sprt(args)
    else:
        run_datagen(args)


if __name__ == "__main__":
    main()
