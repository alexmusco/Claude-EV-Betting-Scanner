#!/usr/bin/env bash
#
# betedge setup. Safe to run any number of times.
#
#   bash setup.sh
#
# Builds the virtual environment, installs dependencies, checks your API
# key, and prints the shell alias with the correct path filled in.
#
# Run it again after every update — new versions sometimes add a
# dependency, and this is what puts it in place.

# Work from the folder this script lives in, whatever directory you ran it
# from. This is the single most common cause of "no such file or directory:
# .venv/bin/activate" — being one folder up, or in a stale copy.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
HERE="$PWD"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; }

echo
bold "betedge setup"
echo "  $HERE"
echo

# --- 1. Is this actually the project folder? -------------------------------

if [ ! -f "requirements.txt" ] || [ ! -d "betedge" ]; then
  fail "This doesn't look like the betedge folder."
  echo "    Expected to find requirements.txt and a betedge/ directory here."
  echo "    Move this script into the project folder and run it again."
  exit 1
fi
ok "found the project"

# --- 2. Python ------------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 is not installed."
  echo "    Run: xcode-select --install"
  echo "    Then run this script again."
  exit 1
fi

PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
  fail "Python $PYV is too old — betedge needs 3.10 or newer."
  echo "    Install a newer Python from python.org, then run this again."
  exit 1
fi
ok "python3 $PYV"

# --- 3. Virtual environment ----------------------------------------------

if [ -x ".venv/bin/python3" ]; then
  ok "virtual environment already here"
else
  if [ -e ".venv" ]; then
    warn "removing a broken .venv"
    rm -rf .venv
  fi
  printf '  … creating the virtual environment\n'
  if ! python3 -m venv .venv; then
    fail "could not create the virtual environment"
    exit 1
  fi
  ok "virtual environment created"
fi

VPY=".venv/bin/python3"

# --- 4. Dependencies ------------------------------------------------------

printf '  … installing dependencies\n'
if ! "$VPY" -m pip install --quiet --upgrade pip 2>/dev/null; then
  warn "could not upgrade pip — carrying on"
fi
if ! "$VPY" -m pip install --quiet -r requirements.txt; then
  fail "dependency install failed"
  echo "    Try it without --quiet to see why:"
  echo "    $VPY -m pip install -r requirements.txt"
  exit 1
fi
ok "dependencies installed"

# --- 5. API key -----------------------------------------------------------

if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example .env
    warn "created .env from the template — put your API key in it"
  else
    printf 'ODDS_API_KEY=\n' > .env
    warn "created an empty .env — put your API key in it"
  fi
elif grep -qE '^ODDS_API_KEY=.+' .env; then
  ok "API key found in .env"
else
  warn ".env has no API key in it"
fi

# --- 6. Database ----------------------------------------------------------

mkdir -p data reports

if [ -s "data/betedge.db" ]; then
  SIZE=$(du -h "data/betedge.db" | cut -f1)
  ok "database here ($SIZE)"
else
  warn "no database in this folder yet — it'll be created on your first scan"
  # A database in another copy of the project is the usual reason bets
  # seem to have vanished after an update.
  OTHERS=$(find "$HOME" -name "betedge.db" -size +8k -not -path "$HERE/*" 2>/dev/null | head -5)
  if [ -n "$OTHERS" ]; then
    echo
    warn "found a database with bets in it somewhere else:"
    echo "$OTHERS" | sed 's/^/      /'
    echo "      To bring your bet history across:"
    echo "      cp '<that path>' '$HERE/data/betedge.db'"
  fi
fi

# --- 7. Verify ------------------------------------------------------------

echo
printf '  … checking the API key works\n'
redact() { sed -E 's/apiKey=[A-Za-z0-9]+/apiKey=***/g'; }

if QUOTA=$("$VPY" -m betedge quota 2>&1); then
  echo "$QUOTA" | redact | sed 's/^/      /'
else
  warn "betedge ran but the API check failed:"
  echo "$QUOTA" | tail -3 | redact | sed 's/^/      /'
fi

# --- 8. Alias -------------------------------------------------------------

ALIAS_LINE="alias bet='cd \"$HERE\" && source .venv/bin/activate && python3 -m betedge'"
SHELL_RC="$HOME/.zshrc"
[ -n "$BASH_VERSION" ] && [ ! -f "$SHELL_RC" ] && SHELL_RC="$HOME/.bashrc"

echo
if grep -qF "alias bet=" "$SHELL_RC" 2>/dev/null; then
  if grep -qF "$HERE" "$SHELL_RC" 2>/dev/null; then
    ok "the 'bet' alias already points here"
  else
    warn "a 'bet' alias exists but points at a different folder."
    echo "    Edit $SHELL_RC and replace it with:"
    echo "    $ALIAS_LINE"
  fi
else
  printf "  Add the 'bet' alias to %s? [y/N] " "$(basename "$SHELL_RC")"
  read -r REPLY
  echo
  case "$REPLY" in
    [yY]*)
      printf '\n%s\n' "$ALIAS_LINE" >> "$SHELL_RC"
      ok "alias added — open a new Terminal window, or run: source $SHELL_RC"
      ;;
    *)
      echo "    Skipped. To add it yourself:"
      echo "    echo \"$ALIAS_LINE\" >> $SHELL_RC"
      ;;
  esac
fi

# --- Done -----------------------------------------------------------------

echo
bold "Ready."
echo
if grep -qE '^ *amount: *1000 *(#.*)?$' config.yaml 2>/dev/null; then
  warn "bankroll.amount in config.yaml is still the 1000 default."
  echo "    Set it to what you are actually willing to lose — every stake"
  echo "    recommendation is a fraction of that number."
  echo
fi
echo "  With the alias, from any folder:"
echo "    bet daily        # the one to run: budgeted scan + shortlist"
echo "    bet quota        # free — what your config costs per run"
echo "    bet budget       # free — credits left and today's allowance"
echo
echo "  Without it, from this folder:"
echo "    source .venv/bin/activate"
echo "    python3 -m betedge daily"
echo
echo "  Safe to run hourly — the budget governor decides what each run"
echo "  may spend. See the README section on the credit budget."
echo
