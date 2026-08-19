"""
simulations.py

Run SCR trials with `model.run_trials` and write the outcomes to a CSV
named ``{name}_results.csv``.  Usage examples are provided in the
``__main__`` block (commented out by default).
"""

from __future__ import annotations

from datetime import datetime  # noqa: F401  (import kept for future use)
from typing import List, Dict, Any

import pandas as pd

from model import run_trials


def simulations_to_results_csv(
    name: str,
    observations: List[Dict[str, Any]],
) -> None:
    """
    Convert trial *observations* into ``{name}_results.csv``.

    Parameters
    ----------
    name : str
        Prefix for the CSV filename.
    observations : list[dict]
        Output from ``run_trials``; each dict must contain keys
        ``match`` (bool) and ``reasoning`` (list[str]).
    """
    rows = []
    for i, out in enumerate(observations, start=1):
        rows.append(
            {
                "trial": i,
                "direction": out["direction"],
                "match": str(out["match"]),
                "A": out["reasoning"].count("A"),
                "B": out["reasoning"].count("B"),
                "C": out["reasoning"].count("C"),
                "D": out["reasoning"].count("D"),
                "E": out["reasoning"].count("E"),
                "F": out["reasoning"].count("F"),
                "nil": out["reasoning"].count("nil"),
                "reasoning": out["reasoning"],
            }
        )

    df = pd.DataFrame(rows)
    csv_name = f"{name}_results.csv"
    df.to_csv(csv_name, index=False)
    print(f"Wrote all results to {csv_name}")


if __name__ == "__main__":
    simulations_to_results_csv(
         "g0_fw_full",
         run_trials(n=2000, structure="chain three",
                    evidence="A", target="C", sd=0.01, direction="forward"),
     )
    simulations_to_results_csv(
         "g0_bw_full",
         run_trials(n=2000, structure="chain three",
                    evidence="C", target="A", sd=0.01, direction="backward"),
     )
    simulations_to_results_csv(
         "g0_fw_short",
         run_trials(n=2000, structure="chain three",
                    evidence="B", target="C", sd=0.01, direction="forward"),
     )
    simulations_to_results_csv(
         "g0_bw_short",
         run_trials(n=2000, structure="chain three",
                    evidence="C", target="B", sd=0.01, direction="backward"),
     )
    simulations_to_results_csv(
         "g1_fw_full",
         run_trials(n=2000, structure="two pyramids",
                    evidence="A", target="C", sd=0.01, direction="forward"),
     )
    simulations_to_results_csv(
         "g1_bw_full",
         run_trials(n=2000, structure="two pyramids",
                    evidence="C", target="A", sd=0.01, direction="backward"),
     )
    simulations_to_results_csv(
         "g1_fw_short",
         run_trials(n=2000, structure="two pyramids",
                    evidence="A", target="B", sd=0.01, direction="forward"),
     )
    simulations_to_results_csv(
         "g1_bw_short",
         run_trials(n=2000, structure="two pyramids",
                    evidence="B", target="A", sd=0.01, direction="backward"),
     )
    simulations_to_results_csv(
         "g2_fw_full",
         run_trials(n=2000, structure="tall glass",
                    evidence="A", target="C", sd=0.01, direction="forward"),
     )
    simulations_to_results_csv(
         "g2_bw_full",
         run_trials(n=2000, structure="tall glass",
                    evidence="C", target="A", sd=0.01, direction="backward"),
     )
    simulations_to_results_csv(
         "g2_fw_short",
         run_trials(n=2000, structure="tall glass",
                    evidence="B", target="C", sd=0.01, direction="forward"),
     )
    simulations_to_results_csv(
         "g2_bw_short",
         run_trials(n=2000, structure="tall glass",
                    evidence="C", target="B", sd=0.01, direction="backward"),
     )
    # Uncomment any of the following lines to (re)generate result files.
    #  simulations_to_results_csv(
    #      "g0_fw_full",
    #      run_trials(n=2000, structure="chain three",
    #                 evidence="A", target="C", sd=0.01, direction="forward"),
    #  )
    #  simulations_to_results_csv(
    #      "g0_bw_full",
    #      run_trials(n=2000, structure="chain three",
    #                 evidence="C", target="A", sd=0.01, direction="backward"),
    #  )
    #  simulations_to_results_csv(
    #      "g0_fw_short",
    #      run_trials(n=2000, structure="chain three",
    #                 evidence="B", target="C", sd=0.01, direction="forward"),
    #  )
    #  simulations_to_results_csv(
    #      "g0_bw_short",
    #      run_trials(n=2000, structure="chain three",
    #                 evidence="C", target="B", sd=0.01, direction="backward"),
    #  )
    #  simulations_to_results_csv(
    #      "g1_fw_full",
    #      run_trials(n=2000, structure="two pyramids",
    #                 evidence="A", target="C", sd=0.01, direction="forward"),
    #  )
    #  simulations_to_results_csv(
    #      "g1_bw_full",
    #      run_trials(n=2000, structure="two pyramids",
    #                 evidence="C", target="A", sd=0.01, direction="backward"),
    #  )
    #  simulations_to_results_csv(
    #      "g1_fw_short",
    #      run_trials(n=2000, structure="two pyramids",
    #                 evidence="B", target="C", sd=0.01, direction="forward"),
    #  )
    #  simulations_to_results_csv(
    #      "g1_bw_short",
    #      run_trials(n=2000, structure="two pyramids",
    #                 evidence="C", target="B", sd=0.01, direction="backward"),
    #  )
    #  simulations_to_results_csv(
    #      "g2_fw_full",
    #      run_trials(n=2000, structure="tall glass",
    #                 evidence="A", target="C", sd=0.01, direction="forward"),
    #  )
    #  simulations_to_results_csv(
    #      "g2_bw_full",
    #      run_trials(n=2000, structure="tall glass",
    #                 evidence="C", target="A", sd=0.01, direction="backward"),
    #  )
    #  simulations_to_results_csv(
    #      "g2_fw_short",
    #      run_trials(n=2000, structure="tall glass",
    #                 evidence="B", target="C", sd=0.01, direction="forward"),
    #  )
    #  simulations_to_results_csv(
    #      "g2_bw_short",
    #      run_trials(n=2000, structure="tall glass",
    #                 evidence="C", target="B", sd=0.01, direction="backward"),
    #  )
    #  simulations_to_results_csv(
    #      "g3_fw_full",
    #      run_trials(n=1000, structure="highway entrance",
    #                 evidence="A", target="C", sd=0.01, direction="forward"),
    #  )
    #  simulations_to_results_csv(
    #      "g3_bw_full",
    #      run_trials(n=1000, structure="highway entrance",
    #                 evidence="C", target="A", sd=0.01, direction="backward"),
    #  )
    #  simulations_to_results_csv(
    #      "g3_fw_short",
    #      run_trials(n=1000, structure="highway entrance",
    #                 evidence="B", target="C", sd=0.01, direction="forward"),
    #  )
    #  simulations_to_results_csv(
    #      "g3_bw_short",
    #      run_trials(n=1000, structure="highway entrance",
    #                 evidence="C", target="B", sd=0.01, direction="backward"),
    #  )

     
     