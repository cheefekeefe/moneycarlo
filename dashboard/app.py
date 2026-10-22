"""MoneyCarlo dashboard: slate overview -> game drill-down -> model track record.

Run with `moneycarlo dashboard` (or `streamlit run dashboard/app.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from moneycarlo import db, evaluate
from moneycarlo.config import DB_PATH

st.set_page_config(page_title="MoneyCarlo NFL", page_icon="🏈", layout="wide")

# Categorical slots 1-2 of the reference palette: home = blue, away = orange.
HOME, AWAY = "#2a78d6", "#eb6834"
NEUTRAL = "#8a8984"
SEQ = [[0, "rgba(42,120,214,0.04)"], [1, "#1c4f8f"]]
STAT_LABELS = {
    "pass_att": "Pass attempts", "pass_cmp": "Completions", "pass_yds": "Passing yards", "pass_td": "Passing TDs",
    "pass_int": "Interceptions", "rush_att": "Carries", "rush_yds": "Rushing yards", "rush_td": "Rushing TDs",
    "targets": "Targets", "receptions": "Receptions", "rec_yds": "Receiving yards", "rec_td": "Receiving TDs",
}


@st.cache_data(ttl=60)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with db.connect(DB_PATH, read_only=True) as con:
        return con.execute(sql, list(params)).df()


def scorecards(kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with db.connect(DB_PATH, read_only=True) as con:
        return evaluate.game_scorecard(con, kind), evaluate.player_scorecard(con, kind)


def fig_layout(fig: go.Figure, height: int = 300, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), bargap=0.08,
                      hoverlabel=dict(namelength=-1), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def spread_label(home: str, away: str, margin: float) -> str:
    """Home margin -> conventional favourite spread, e.g. 'KC -3.5'."""
    fav = home if margin >= 0 else away
    return f"{fav} -{abs(margin):.1f}"


if not DB_PATH.exists():
    st.title("MoneyCarlo NFL")
    st.info("No predictions yet. Run `moneycarlo predict` (and optionally `moneycarlo backtest`) first.")
    st.stop()

try:
    runs = query("SELECT DISTINCT kind, season, week FROM runs ORDER BY season DESC, week DESC")
except Exception as exc:  # DB locked by a running CLI command
    st.warning(f"Database is busy ({exc}). Refresh in a moment.")
    st.stop()
if runs.empty:
    st.info("No predictions yet. Run `moneycarlo predict` first.")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🏈 MoneyCarlo")
    kinds = runs["kind"].unique().tolist()
    kind = st.radio("Predictions", kinds, index=kinds.index("live") if "live" in kinds else 0,
                    format_func=lambda k: {"live": "Live (pre-game)", "backtest": "Backtest"}[k])
    r = runs[runs["kind"] == kind]
    season = st.selectbox("Season", sorted(r["season"].unique(), reverse=True))
    weeks = sorted(r.loc[r["season"] == season, "week"].unique(), reverse=True)
    week = st.selectbox("Week", weeks, format_func=lambda w: f"Week {w}")
    st.caption("Each game is simulated thousands of times drive by drive; every number here "
               "is a summary of those simulations. Ranges are 10th-90th percentiles.")

latest = query(
    """SELECT run_id, created_at, n_sims FROM runs WHERE kind = ? AND season = ? AND week = ?
       ORDER BY created_at DESC LIMIT 1""", (kind, int(season), int(week)))
run_id = latest["run_id"].iloc[0]
games = query("SELECT * FROM game_predictions WHERE run_id = ? ORDER BY gameday, gametime, game_id", (run_id,))
results = query("SELECT game_id, home_score, away_score FROM game_results")
games = games.merge(results, on="game_id", how="left")

tab_slate, tab_game, tab_perf = st.tabs(["📋 Slate", "🔎 Game detail", "📈 Model performance"])

# ---------------------------------------------------------------- slate
with tab_slate:
    st.subheader(f"{season} · Week {week}")
    st.caption(f"Run {run_id} · {latest['created_at'].iloc[0]:%Y-%m-%d %H:%M} UTC · "
               f"{int(latest['n_sims'].iloc[0]):,} simulations per game")

    done = games.dropna(subset=["home_score"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games", len(games))
    c2.metric("Avg projected total", f"{games['total_mean'].mean():.1f}")
    if len(done):
        hits = ((done["home_win_prob"] > 0.5) == (done["home_score"] > done["away_score"])).mean()
        c3.metric("Winners called", f"{hits:.0%}", help=f"{len(done)} completed games")
        c4.metric("Margin error (MAE)", f"{(done['margin_mean'] - (done['home_score'] - done['away_score'])).abs().mean():.1f}")

    table = pd.DataFrame({
        "Kickoff": pd.to_datetime(games["gameday"]).dt.strftime("%a %b %d") + " " + games["gametime"].fillna(""),
        "Matchup": games["away_team"] + " @ " + games["home_team"],
        "Projected score": games.apply(
            lambda g: f"{g.away_team} {g.away_pts_mean:.1f} – {g.home_team} {g.home_pts_mean:.1f}", axis=1),
        "Winner": games.apply(lambda g: g.home_team if g.home_win_prob >= g.away_win_prob else g.away_team, axis=1),
        "Win prob": games[["home_win_prob", "away_win_prob"]].max(axis=1) * 100,
        "Spread": games.apply(lambda g: spread_label(g.home_team, g.away_team, g.margin_mean), axis=1),
        "Total": games["total_mean"],
        "Total range": games.apply(lambda g: f"{g.total_p10:.0f}–{g.total_p90:.0f}", axis=1),
        "Most likely": games.apply(lambda g: f"{g.away_team} {g.most_likely_score.split('-')[1]}–"
                                             f"{g.most_likely_score.split('-')[0]} {g.home_team}", axis=1),
        "Final": games.apply(lambda g: "" if pd.isna(g.home_score)
                             else f"{g.away_team} {int(g.away_score)}–{int(g.home_score)} {g.home_team}", axis=1),
        "✓": games.apply(lambda g: "" if pd.isna(g.home_score) or g.home_score == g.away_score
                         else ("✅" if (g.home_win_prob > 0.5) == (g.home_score > g.away_score) else "❌"), axis=1),
    })
    st.dataframe(
        table, hide_index=True, use_container_width=True, height=36 * (len(table) + 1) + 3,
        column_config={
            "Win prob": st.column_config.ProgressColumn("Win prob", format="%.0f%%", min_value=50, max_value=100),
            "Total": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.markdown("##### Win probability")
    g2 = games.iloc[::-1]
    labels = g2["away_team"] + " @ " + g2["home_team"]
    fig = go.Figure()
    fig.add_bar(y=labels, x=g2["away_win_prob"], orientation="h", name="Away", marker_color=AWAY,
                text=g2["away_team"] + " " + (g2["away_win_prob"] * 100).round().astype(int).astype(str) + "%",
                textposition="inside", insidetextanchor="start",
                hovertemplate="%{y}<br>Away win %{x:.1%}<extra></extra>")
    fig.add_bar(y=labels, x=g2["home_win_prob"], orientation="h", name="Home", marker_color=HOME,
                text=g2["home_team"] + " " + (g2["home_win_prob"] * 100).round().astype(int).astype(str) + "%",
                textposition="inside", insidetextanchor="end",
                hovertemplate="%{y}<br>Home win %{x:.1%}<extra></extra>")
    fig.add_vline(x=0.5, line_width=1, line_dash="dot", line_color=NEUTRAL)
    fig.update_traces(marker_line_width=0, textfont_color="white")
    fig_layout(fig, height=40 + 34 * len(games), barmode="stack",
               legend=dict(orientation="h", y=1.05, x=0), xaxis=dict(tickformat=".0%", range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- game detail
with tab_game:
    opts = games["game_id"].tolist()
    gid = st.selectbox("Game", opts, format_func=lambda i: (
        lambda g: f"{g.away_team} @ {g.home_team} — {pd.to_datetime(g.gameday):%a %b %d}")(
        games.set_index("game_id").loc[i]))
    g = games.set_index("game_id").loc[gid]
    home, away = g["home_team"], g["away_team"]

    st.subheader(f"{away} @ {home}")
    m = st.columns(5)
    tiles = [
        (f"{away} projected", f"{g.away_pts_mean:.1f}", f"range {g.away_pts_p10:.0f}–{g.away_pts_p90:.0f}"),
        (f"{home} projected", f"{g.home_pts_mean:.1f}", f"range {g.home_pts_p10:.0f}–{g.home_pts_p90:.0f}"),
        ("Win probability", f"{home} {g.home_win_prob:.0%}", f"{away} {g.away_win_prob:.0%}"
         + (f" · tie {g.tie_prob:.1%}" if g.tie_prob >= 0.001 else "")),
        ("Spread", spread_label(home, away, g.margin_mean),
         f"{home} margin {g.margin_p10:+.0f} to {g.margin_p90:+.0f}"),
        ("Total", f"{g.total_mean:.1f}", f"range {g.total_p10:.0f}–{g.total_p90:.0f}"),
    ]
    for col, (label, value, sub) in zip(m, tiles):
        col.metric(label, value)
        col.caption(sub)
    if not pd.isna(g["home_score"]):
        st.success(f"Final: {away} {int(g.away_score)} – {int(g.home_score)} {home}")

    dist = query("SELECT home_pts, away_pts, prob FROM score_distributions WHERE run_id = ? AND game_id = ?",
                 (run_id, gid))
    dist["margin"] = dist["home_pts"] - dist["away_pts"]
    dist["total"] = dist["home_pts"] + dist["away_pts"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### Final margin ({home} perspective)")
        md = dist.groupby("margin")["prob"].sum().reset_index()
        md = md[md["margin"].between(-40, 40)]
        fig = go.Figure(go.Bar(
            x=md["margin"], y=md["prob"],
            marker_color=np.where(md["margin"] > 0, HOME, np.where(md["margin"] < 0, AWAY, NEUTRAL)),
            marker_line_width=0,
            hovertemplate="Margin %{x:+d}<br>%{y:.1%}<extra></extra>"))
        if not pd.isna(g["home_score"]):
            fig.add_vline(x=g.home_score - g.away_score, line_width=2, line_color=NEUTRAL,
                          annotation_text="actual", annotation_position="top")
        fig_layout(fig, yaxis=dict(tickformat=".0%"), xaxis_title=f"← {away} wins | {home} wins →")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("##### Combined total")
        td = dist.groupby("total")["prob"].sum().reset_index()
        fig = go.Figure(go.Bar(x=td["total"], y=td["prob"], marker_color=HOME, marker_line_width=0,
                               hovertemplate="Total %{x}<br>%{y:.1%}<extra></extra>"))
        if not pd.isna(g["home_score"]):
            fig.add_vline(x=g.home_score + g.away_score, line_width=2, line_color=NEUTRAL,
                          annotation_text="actual", annotation_position="top")
        fig_layout(fig, yaxis=dict(tickformat=".0%"), xaxis_title="Points")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Most likely final scores", expanded=False):
        top = dist.sort_values("prob", ascending=False).head(15)
        st.dataframe(pd.DataFrame({
            "Score": away + " " + top["away_pts"].astype(str) + " – " + top["home_pts"].astype(str) + " " + home,
            "Probability": top["prob"],
        }), hide_index=True, column_config={"Probability": st.column_config.NumberColumn(format="percent")})
        grid = dist[(dist["home_pts"] <= 50) & (dist["away_pts"] <= 50)]
        heat = grid.pivot_table(index="away_pts", columns="home_pts", values="prob", aggfunc="sum").fillna(0)
        fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale=SEQ,
                                   hovertemplate=f"{home} %{{x}} – {away} %{{y}}<br>%{{z:.2%}}<extra></extra>",
                                   colorbar=dict(tickformat=".1%")))
        fig_layout(fig, height=420, xaxis_title=f"{home} points", yaxis_title=f"{away} points")
        st.plotly_chart(fig, use_container_width=True)

    # ---- team box
    st.markdown("##### Team projections")
    tp = query("SELECT team, stat, mean, p10, p90 FROM team_predictions WHERE run_id = ? AND game_id = ?", (run_id, gid))
    tr = query("SELECT team, stat, value FROM team_results WHERE game_id = ?", (gid,))
    names = {"points": "Points", "plays": "Offensive plays", "attempts": "Pass attempts", "completions": "Completions",
             "pass_yards": "Passing yards", "pass_td": "Passing TDs", "interceptions": "Interceptions",
             "sacks": "Sacks taken", "carries": "Rush attempts", "rush_yards": "Rushing yards", "rush_td": "Rushing TDs",
             "fumbles_lost": "Fumbles lost"}
    rows = []
    for stat, label in names.items():
        row = {"Stat": label}
        for team in (away, home):
            x = tp[(tp["team"] == team) & (tp["stat"] == stat)]
            if x.empty:
                continue
            x = x.iloc[0]
            row[team] = f"{x['mean']:.1f}  ({x.p10:.0f}–{x.p90:.0f})"
            a = tr[(tr["team"] == team) & (tr["stat"] == stat)]
            if not a.empty:
                row[f"{team} actual"] = f"{a['value'].iloc[0]:.0f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=36 * (len(rows) + 1) + 3)

    # ---- players
    st.markdown("##### Player projections")
    st.caption("Mean projection with the 10th–90th percentile range. TD % = chance of at least one.")
    pp = query("SELECT * FROM player_predictions WHERE run_id = ? AND game_id = ?", (run_id, gid))
    pr = query("SELECT player_id, stat, value FROM player_results WHERE game_id = ?", (gid,))
    is_final = not pd.isna(g["home_score"])

    def stat_table(team: str, stats: dict[str, str], td_stat: str, sort_by: str, positions=None) -> pd.DataFrame:
        d = pp[(pp["team"] == team) & pp["stat"].isin(stats)]
        if positions:
            d = d[d["position"].isin(positions)]
        if d.empty:
            return pd.DataFrame()
        out = d.pivot_table(index=["player_id", "player_name", "position"], columns="stat",
                            values=["mean", "p10", "p90", "prob_pos"], aggfunc="first")
        res = pd.DataFrame(index=out.index)
        for s, label in stats.items():
            if ("mean", s) not in out.columns:
                continue
            res[label] = out[("mean", s)].round(1)
            if s != td_stat:
                res[f"{label} range"] = (out[("p10", s)].round().astype(int).astype(str) + "–"
                                         + out[("p90", s)].round().astype(int).astype(str))
        if ("prob_pos", td_stat) in out.columns:
            res["TD %"] = out[("prob_pos", td_stat)] * 100
        if is_final:
            act = pr.pivot_table(index="player_id", columns="stat", values="value", aggfunc="first")
            for s, label in stats.items():
                ids = res.index.get_level_values("player_id")
                res[f"{label} (actual)"] = act[s].reindex(ids).fillna(0).to_numpy() if s in act.columns else 0
        res = res.sort_values(stats[sort_by], ascending=False).reset_index().drop(columns="player_id")
        return res.rename(columns={"player_name": "Player", "position": "Pos"})

    cfg = {"TD %": st.column_config.NumberColumn(format="%.0f%%")}
    for team in (away, home):
        st.markdown(f"**{team}**")
        passing = stat_table(team, {"pass_cmp": "Cmp", "pass_att": "Att", "pass_yds": "Pass yds",
                                    "pass_td": "Pass TD", "pass_int": "INT"}, "pass_td", "pass_att")
        rushing = stat_table(team, {"rush_att": "Carries", "rush_yds": "Rush yds", "rush_td": "Rush TD"},
                             "rush_td", "rush_yds")
        receiving = stat_table(team, {"targets": "Targets", "receptions": "Rec", "rec_yds": "Rec yds",
                                      "rec_td": "Rec TD"}, "rec_td", "rec_yds", positions=["WR", "TE", "RB", "FB"])
        t1, t2, t3 = st.tabs(["Passing", "Rushing", "Receiving"])
        with t1:
            st.dataframe(passing, hide_index=True, use_container_width=True, column_config=cfg)
        with t2:
            st.dataframe(rushing[rushing["Carries"] >= 0.5] if not rushing.empty else rushing,
                         hide_index=True, use_container_width=True, column_config=cfg)
        with t3:
            st.dataframe(receiving[receiving["Targets"] >= 0.5] if not receiving.empty else receiving,
                         hide_index=True, use_container_width=True, column_config=cfg)

# ---------------------------------------------------------------- performance
with tab_perf:
    card, pcard = scorecards(kind)
    st.subheader(f"Track record · {'live pre-game' if kind == 'live' else 'backtest'} predictions")
    if card.empty:
        st.info("No completed games with stored predictions yet. Results attach automatically after "
                "`moneycarlo update` once games are played. Run `moneycarlo backtest` for a historical track record.")
        st.stop()
    seasons = sorted(card["season"].unique(), reverse=True)
    pick = st.multiselect("Seasons", seasons, default=seasons)
    card = card[card["season"].isin(pick)]
    pcard = pcard[pcard["game_id"].isin(card["game_id"])] if not pcard.empty else pcard
    s = evaluate.summary(card)

    m = st.columns(7)
    m[0].metric("Games", s["games"])
    m[1].metric("Winners called", f"{s['winner_accuracy']:.1%}")
    m[2].metric("Brier score", f"{s['brier']:.3f}", help="Mean squared error of win probability. 0.25 = coin flip; lower is better.")
    m[3].metric("Margin MAE", f"{s['margin_mae']:.1f}")
    m[4].metric("Total MAE", f"{s['total_mae']:.1f}")
    cover_help = "Share of actual results inside the predicted 10th–90th percentile range. Well calibrated ≈ 80%."
    m[5].metric("Margin in range", f"{s['margin_80_coverage']:.0%}", help=cover_help)
    m[6].metric("Total in range", f"{s['total_80_coverage']:.0%}", help=cover_help)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Win-probability calibration")
        st.caption("When the model says 70%, the favourite should win about 70% of the time.")
        fav_p = card[["p_home"]].assign(fp=np.maximum(card["p_home"], 1 - card["p_home"]),
                                        fw=np.where(card["p_home"] >= 0.5, card["home_won"], 1 - card["home_won"]))
        bins = pd.cut(fav_p["fp"], [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0], include_lowest=True)
        cal = fav_p.groupby(bins, observed=True).agg(pred=("fp", "mean"), actual=("fw", "mean"), n=("fp", "size")).reset_index()
        fig = go.Figure()
        fig.add_scatter(x=[0.5, 1], y=[0.5, 1], mode="lines", line=dict(color=NEUTRAL, dash="dot", width=1),
                        name="Perfect", hoverinfo="skip")
        fig.add_scatter(x=cal["pred"], y=cal["actual"], mode="lines+markers", name="Model",
                        line=dict(color=HOME, width=2), marker=dict(size=10, line=dict(width=2, color="white")),
                        customdata=cal["n"],
                        hovertemplate="Predicted %{x:.0%}<br>Actual %{y:.0%}<br>%{customdata} games<extra></extra>")
        fig_layout(fig, xaxis=dict(tickformat=".0%", title="Predicted favourite win %"),
                   yaxis=dict(tickformat=".0%", title="Actual"), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("##### Accuracy by week")
        wk = card.groupby(["season", "week"]).agg(acc=("correct", "mean"), mae=("margin_err", lambda e: e.abs().mean()),
                                                  n=("game_id", "size")).reset_index()
        wk["label"] = wk["season"].astype(str) + " W" + wk["week"].astype(str)
        fig = go.Figure(go.Bar(x=wk["label"], y=wk["acc"], marker_color=HOME, marker_line_width=0,
                               customdata=np.stack([wk["n"], wk["mae"]], axis=1),
                               hovertemplate="%{x}<br>Winners %{y:.0%}<br>%{customdata[0]} games"
                                             "<br>Margin MAE %{customdata[1]:.1f}<extra></extra>"))
        fig.add_hline(y=s["winner_accuracy"], line_dash="dot", line_color=NEUTRAL,
                      annotation_text=f"avg {s['winner_accuracy']:.0%}", annotation_position="top left")
        fig_layout(fig, yaxis=dict(tickformat=".0%", range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Player stat accuracy")
    st.caption("Bias = average (projection − actual). Coverage = share of actuals inside the 10th–90th "
               "percentile range (target ≈ 80%; TD and INT counts are discrete so run higher).")
    ps = evaluate.player_summary(pcard)
    if not ps.empty:
        ps["stat"] = ps["stat"].map(STAT_LABELS).fillna(ps["stat"])
        ps["coverage_80"] *= 100
        st.dataframe(ps.rename(columns={"stat": "Stat", "n": "Projections", "mae": "MAE", "bias": "Bias",
                                        "coverage_80": "80% coverage"}),
                     hide_index=True, use_container_width=True, height=36 * (len(ps) + 1) + 3,
                     column_config={"MAE": st.column_config.NumberColumn(format="%.2f"),
                                    "Bias": st.column_config.NumberColumn(format="%+.2f"),
                                    "80% coverage": st.column_config.NumberColumn(format="%.0f%%")})

    with st.expander("All graded games"):
        st.dataframe(card[["season", "week", "away_team", "home_team", "away_pts_mean", "home_pts_mean",
                           "away_score", "home_score", "home_win_prob", "margin_mean", "actual_margin",
                           "total_mean", "actual_total"]].sort_values(["season", "week"], ascending=False),
                     hide_index=True, use_container_width=True)
