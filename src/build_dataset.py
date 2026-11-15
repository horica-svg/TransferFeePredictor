import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_advanced_dataset():
    print("1. Loading CSV files...")
    players = pd.read_csv(RAW_DIR / "players.csv")
    appearances = pd.read_csv(
        RAW_DIR / "appearances.csv",
        usecols=["appearance_id", "player_id", "date", "competition_id", "goals", "assists", "minutes_played"],
    )
    clubs = pd.read_csv(RAW_DIR / "clubs.csv", usecols=["club_id", "domestic_competition_id"])
    lineups = pd.read_csv(
        RAW_DIR / "game_lineups.csv", usecols=["date", "game_id", "player_id", "type", "team_captain"]
    )
    events = pd.read_csv(
        RAW_DIR / "game_events.csv", usecols=["player_id", "type", "description"]
    )

    # 1. Filtrare target valid + doar jucatori activi (ultimele 3 sezoane)
    players = players.dropna(subset=["market_value_in_eur"])
    players = players[players["market_value_in_eur"] > 0]
    players = players[players["last_season"] >= players["last_season"].max() - 2]

    # Data de referinta
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    ref_date = appearances["date"].max()
    if pd.isna(ref_date):
        ref_date = pd.Timestamp.now()
    one_year_ago = ref_date - pd.Timedelta(days=365)

    # 2. Varsta si Durata Contractuala
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    players["age"] = (ref_date - players["date_of_birth"]).dt.days / 365.25

    players["contract_expiration_date"] = pd.to_datetime(players["contract_expiration_date"], errors="coerce")
    # NaN ramane NaN (LightGBM il trateaza nativ); contractul expirat devine 0
    players["contract_remaining_months"] = ((players["contract_expiration_date"] - ref_date).dt.days / 30.44).clip(lower=0)

    # 3. Procesare Lineups (Titularizari si Capitan)
    print("2. Processing data from game_lineups...")
    lineups["date"] = pd.to_datetime(lineups["date"], errors="coerce")

    # Flag titular (type este de regula 'starting_lineup' sau 'substitutes')
    lineups["is_starter"] = lineups["type"].str.contains("starting", case=False, na=False).astype(int)
    
    lineups["is_captain"] = lineups["team_captain"].fillna(0).astype(int)

    # Agregare lineups ultimul an
    recent_lineups = lineups[lineups["date"] >= one_year_ago]
    lineup_stats = recent_lineups.groupby("player_id").agg(
        lineup_apps_recent=("game_id", "count"),
        starts_recent=("is_starter", "sum"),
        captain_games_recent=("is_captain", "sum")
    ).reset_index()

    lineup_stats["starter_ratio_recent"] = np.where(
        lineup_stats["lineup_apps_recent"] > 0,
        lineup_stats["starts_recent"] / lineup_stats["lineup_apps_recent"],
        0.0
    )

    # 4. Procesare Game Events (Penalty-uri, Goluri decisive, Eliminari)
    print("3. Processing events from game_events...")
    # Coloana type indica tipul: 'Goals', 'Cards', 'Substitutions' etc.
    events["description_clean"] = events["description"].fillna("").str.lower()
    
    # Penalty-uri marcate
    events["is_penalty_goal"] = (
        (events["type"].str.contains("Goal", case=False, na=False)) & 
        (events["description_clean"].str.contains("penalty", na=False))
    ).astype(int)

    # Cartonas rosu direct sau dublu galben
    events["is_red_card"] = (
        (events["type"].str.contains("Card", case=False, na=False)) & 
        (events["description_clean"].str.contains("red|second yellow", na=False))
    ).astype(int)

    events_stats = events.groupby("player_id").agg(
        total_penalty_goals=("is_penalty_goal", "sum"),
        total_red_cards=("is_red_card", "sum")
    ).reset_index()

    # 5. Agregare Appearances (Cariera + Ultimul An)
    print("4. Processing metrics from appearances...")
    career_stats = appearances.groupby("player_id").agg(
        career_games=("appearance_id", "count"),
        career_goals=("goals", "sum"),
        career_assists=("assists", "sum"),
        career_minutes=("minutes_played", "sum")
    ).reset_index()

    recent_apps = appearances[appearances["date"] >= one_year_ago]
    recent_stats = recent_apps.groupby("player_id").agg(
        recent_minutes=("minutes_played", "sum"),
        recent_goals=("goals", "sum"),
        recent_assists=("assists", "sum")
    ).reset_index()

    recent_stats["recent_90s"] = recent_stats["recent_minutes"] / 90.0
    recent_stats["goals_per_90"] = np.where(
        recent_stats["recent_90s"] >= 1.0,
        recent_stats["recent_goals"] / recent_stats["recent_90s"],
        0.0
    )
    recent_stats["assists_per_90"] = np.where(
        recent_stats["recent_90s"] >= 1.0,
        recent_stats["recent_assists"] / recent_stats["recent_90s"],
        0.0
    )

    # Meciuri continentale
    continental_comps = ["CL", "EL", "ECL", "USC"]
    appearances["is_continental"] = appearances["competition_id"].isin(continental_comps).astype(int)
    intl_matches = appearances.groupby("player_id")["is_continental"].sum().reset_index(name="continental_matches")

    # 6. Unire tabele
    print("5. Merging and final cleaning...")
    df = players.merge(career_stats, on="player_id", how="left")
    df = df.merge(recent_stats, on="player_id", how="left")
    df = df.merge(lineup_stats, on="player_id", how="left")
    df = df.merge(events_stats, on="player_id", how="left")
    df = df.merge(intl_matches, on="player_id", how="left")

    # Calcul goluri din actiune deschisa (fara penalty)
    df["open_play_goals"] = np.maximum(0, df["career_goals"].fillna(0) - df["total_penalty_goals"].fillna(0))

    # Adaugare club & liga
    clubs_subset = clubs.rename(columns={"domestic_competition_id": "league_id"})
    df = df.merge(clubs_subset, left_on="current_club_id", right_on="club_id", how="left")

    # Umplere valori lipsa: zero doar pentru contori (fara activitate = 0); ratele raman NaN
    numeric_cols = [
        "career_games", "career_goals", "career_assists", "career_minutes",
        "recent_minutes", "starts_recent",
        "captain_games_recent", "total_penalty_goals",
        "open_play_goals", "total_red_cards", "continental_matches"
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # 7. Selectare set final de trasaturi
    final_cols = [
        "player_id", "name", "position", "sub_position", "foot", "height_in_cm",
        "age", "country_of_citizenship", "league_id", "contract_remaining_months",
        "career_minutes", "career_goals", "open_play_goals", "career_assists",
        "recent_minutes", "goals_per_90", "assists_per_90",
        "starter_ratio_recent", "captain_games_recent", "total_red_cards",
        "continental_matches",
        "market_value_in_eur"  # Target
    ]

    final_df = df[[c for c in final_cols if c in df.columns]].dropna(subset=["age"])

    final_df = final_df.reset_index(drop=True)
    
    output_path = OUT_DIR / "dataset_processed.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved file: {output_path}")
    print(f"Rows: {len(final_df)} | Columns: {len(final_df.columns)}")

if __name__ == "__main__":
    build_advanced_dataset()