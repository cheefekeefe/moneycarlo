"""Command line entry point: `moneycarlo <command>` (or `python -m moneycarlo <command>`)."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from moneycarlo import db, evaluate
from moneycarlo.config import TRAIN_SEASONS, SimParams


def _params(con, args) -> SimParams:
    stored = db.latest_params(con)
    p = SimParams.from_dict(stored) if stored else SimParams()
    if getattr(args, "sims", None):
        p.n_sims = args.sims
    return p


def cmd_update(args):
    from moneycarlo.nfl import data, pipeline

    cur = data.current_season()
    data.update(list(range(cur - TRAIN_SEASONS, cur + 1)))
    with db.connect() as con:
        ctx = pipeline.build_context(cur, _params(con, args))
        n = pipeline.sync_results(con, ctx)
    print(f"Data refreshed through {cur}; {n} completed games stored as results.")


def cmd_predict(args):
    from moneycarlo.nfl import data, pipeline

    season = args.season or data.current_season()
    if not args.no_update:
        data.update(list(range(season - TRAIN_SEASONS, season + 1)))
    with db.connect() as con:
        ctx = pipeline.build_context(season, _params(con, args))
        pipeline.sync_results(con, ctx)
        week = args.week or pipeline.next_week(ctx.schedules, season)
        run_id = pipeline.run_week(con, ctx, season, week, kind="live", seed=args.seed)
        games = con.execute(
            "SELECT away_team, home_team, away_pts_mean, home_pts_mean, home_win_prob, margin_mean, total_mean "
            "FROM game_predictions WHERE run_id = ? ORDER BY gameday, gametime", [run_id]
        ).df()
    print(f"\n{season} week {week}  (run {run_id}, {ctx.params.n_sims:,} sims per game)\n")
    for g in games.itertuples():
        fav, p = (g.home_team, g.home_win_prob) if g.home_win_prob >= 0.5 else (g.away_team, 1 - g.home_win_prob)
        print(f"  {g.away_team:>3} {g.away_pts_mean:5.1f} @ {g.home_team:<3} {g.home_pts_mean:5.1f}   "
              f"{fav} {p:5.1%}   home margin {g.margin_mean:+5.1f}   total {g.total_mean:5.1f}")
    print("\nOpen the dashboard with: moneycarlo dashboard")


def cmd_backtest(args):
    from moneycarlo.nfl import data, pipeline

    seasons = args.seasons or [data.current_season() - 1]
    with db.connect() as con:
        ctx = pipeline.build_context(max(seasons), _params(con, args), first_season=min(seasons) - TRAIN_SEASONS)
        pipeline.sync_results(con, ctx)
        weeks = pipeline.completed_weeks(ctx.schedules, seasons)
        for season, week in weeks:
            pipeline.run_week(con, ctx, season, week, kind="backtest",
                              with_players=not args.no_players, seed=args.seed)
            print(f"  backtested {season} week {week}", flush=True)
        s = evaluate.summary(evaluate.game_scorecard(con, "backtest"))
    _print_summary("Backtest", s)


def cmd_calibrate(args):
    from itertools import product

    from moneycarlo.nfl import data, pipeline

    cur = data.current_season()
    seasons = args.seasons or [cur - 2, cur - 1, cur]
    grid = [{"form_sigma": s, "script_kappa": k}
            for s, k in product(args.sigmas or [0.0, 0.15, 0.3], args.kappas or [0.0, 0.1, 0.2, 0.3])]
    with db.connect() as con:
        params = _params(con, args)
        ctx = pipeline.build_context(max(seasons), params, first_season=min(seasons) - TRAIN_SEASONS)
        weeks = pipeline.completed_weeks(ctx.schedules, seasons)
        best, table = pipeline.calibrate(ctx, weeks, grid)
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        for k, v in best.items():
            setattr(params, k, v)
        metrics = table.loc[table["score"].idxmin()].to_dict()
        db.save_params(con, params.to_dict(), {k: float(v) for k, v in metrics.items()})
    print(f"\nSaved {best}; future predictions will use these.")


def cmd_evaluate(args):
    with db.connect() as con:
        _print_summary(f"{args.kind} predictions", evaluate.summary(evaluate.game_scorecard(con, args.kind)))
        ps = evaluate.player_summary(evaluate.player_scorecard(con, args.kind))
    if not ps.empty:
        print("\nPlayer stats:")
        print(ps.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def cmd_dashboard(args):
    app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)


def _print_summary(title: str, s: dict) -> None:
    if not s:
        print(f"{title}: no completed games with predictions yet.")
        return
    print(f"\n{title}: {s['games']} games")
    print(f"  winner accuracy   {s['winner_accuracy']:.1%}")
    print(f"  brier / log loss  {s['brier']:.3f} / {s['log_loss']:.3f}")
    print(f"  margin MAE        {s['margin_mae']:.2f}   (80% interval coverage {s['margin_80_coverage']:.0%})")
    print(f"  total MAE         {s['total_mae']:.2f}   (80% interval coverage {s['total_80_coverage']:.0%})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moneycarlo", description="NFL Monte Carlo prediction engine")
    ap.add_argument("-q", "--quiet", action="store_true", help="hide progress messages")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("update", help="download latest data and store results").set_defaults(func=cmd_update)

    p = sub.add_parser("predict", help="simulate a week of games")
    p.add_argument("--season", type=int)
    p.add_argument("--week", type=int, help="default: next week with unplayed games")
    p.add_argument("--sims", type=int, help="simulations per game")
    p.add_argument("--seed", type=int)
    p.add_argument("--no-update", action="store_true", help="skip data refresh")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("backtest", help="walk-forward predictions for past weeks")
    p.add_argument("--seasons", type=int, nargs="+")
    p.add_argument("--sims", type=int)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-players", action="store_true")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("calibrate", help="tune simulation dispersion against past results")
    p.add_argument("--seasons", type=int, nargs="+")
    p.add_argument("--sigmas", type=float, nargs="+", help="form shock values to try")
    p.add_argument("--kappas", type=float, nargs="+", help="game-script feedback values to try")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("evaluate", help="score stored predictions against results")
    p.add_argument("--kind", choices=["live", "backtest"], default="live")
    p.set_defaults(func=cmd_evaluate)

    sub.add_parser("dashboard", help="launch the Streamlit dashboard").set_defaults(func=cmd_dashboard)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    # Progress messages from this package only (third-party libraries stay quiet).
    logging.getLogger("moneycarlo").setLevel(logging.WARNING if args.quiet else logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
