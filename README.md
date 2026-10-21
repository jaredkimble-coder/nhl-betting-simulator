# NHL Betting Simulator

RotoWire's NHL betting simulator: every bet is priced by how often it happens across 20,000 simulated games, using sportsbook prices, lineups, MoneyPuck and ShotPropz data.

Live page: https://jaredkimble-coder.github.io/nhl-betting-simulator/

## How it refreshes
n8n pulls odds, lineups and stats on its own schedule and saves each matchup's input. A GitHub Action (`.github/workflows/refresh.yml`) runs every hour:

1. `refresh.py` fetches the saved inputs from the n8n endpoint.
2. It simulates only the games that are new or whose inputs changed (everything else comes from the cache), adds line movement, and builds the page.
3. If anything changed, the page is deployed to GitHub Pages (as a Pages artifact, so the 15 MB page is never committed).

The cache, price history and pre-game predictions live on a separate `state` branch that is overwritten each run, so this repo's history stays small.

## Setup
- Repository secret `N8N_INPUTS_TOKEN`: the token for the n8n inputs endpoint.
- Settings, Pages, Source: GitHub Actions.
- Run it by hand any time: Actions, Refresh NHL betting simulator, Run workflow (tick "force" to rebuild everything).

`engine/` is the simulation (the model, the matchup analysis, the grader). `engine/templates/` holds the page template and the RotoWire logo.
