# Tracker template

`betedge export <path>.xlsx` fills a copy of your existing Excel tracker
from the database rather than writing a CSV. It needs the workbook to copy
from, and that workbook is yours — it is not shipped here, because the
formulas on its three sheets are the thing being preserved.

Put it at:

    betedge/templates/tracker_template.xlsx

or point at it per-run:

    betedge export ~/Desktop/AlexBetTracker.xlsx --template ~/Desktop/blank_tracker.xlsx

The sheet it writes into must be named `1. Bet Entry`, with the column
layout `tracker.py` documents at the top of the file. Only the input
columns are written; every formula is left untouched.
