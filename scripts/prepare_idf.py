"""
Prepare the baseline IDF for our hackathon simulation.

Modifies the DOE reference small office in three ways:
  1. Enable weather-file RunPeriod (default is disabled).
  2. Add a July 1-7 RunPeriod for our Bangalore simulation.
  3. Declare Output:Variable entries so the LLM sensor tools have data.

Original IDF is backed up to baseline_small_office.original.idf on first run.
Subsequent runs restore from that backup, so this script is idempotent
and always produces the same output regardless of prior state.
"""

import sys
import shutil
from pathlib import Path

# Make config importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from eppy.modeleditor import IDF

# Point eppy at EnergyPlus's IDD (schema definition file)
IDD_PATH = config.ENERGYPLUS_DIR / "Energy+.idd"
if not IDD_PATH.exists():
    print(f"[FAIL] IDD not found at {IDD_PATH}")
    sys.exit(1)
IDF.setiddname(str(IDD_PATH))

SOURCE_IDF = config.BASELINE_IDF
BACKUP_IDF = config.IDF_DIR / "baseline_small_office.original.idf"


def prepare():
    # Backup on first run; restore from backup on subsequent runs
    if not BACKUP_IDF.exists():
        shutil.copy(SOURCE_IDF, BACKUP_IDF)
        print(f"[OK] Backed up original: {BACKUP_IDF.name}")
    else:
        shutil.copy(BACKUP_IDF, SOURCE_IDF)
        print(f"[OK] Restored from backup (starting clean)")

    idf = IDF(str(SOURCE_IDF))

    # -----------------------------------------------------------------------
    # 1. Enable weather-file RunPeriod simulation
    # -----------------------------------------------------------------------
    for sc in idf.idfobjects["SIMULATIONCONTROL"]:
        sc.Run_Simulation_for_Weather_File_Run_Periods = "Yes"
        sc.Run_Simulation_for_Sizing_Periods = "Yes"  # keep sizing for HVAC
        print(f"[OK] SimulationControl: weather file simulation enabled")

    # -----------------------------------------------------------------------
    # 2. Set RunPeriod to our July week
    # -----------------------------------------------------------------------
    for rp in list(idf.idfobjects["RUNPERIOD"]):
        idf.removeidfobject(rp)

    idf.newidfobject(
        "RUNPERIOD",
        Name="Bangalore_July_Week",
        Begin_Month=config.SIMULATION_START_MONTH,
        Begin_Day_of_Month=config.SIMULATION_START_DAY,
        End_Month=config.SIMULATION_END_MONTH,
        End_Day_of_Month=config.SIMULATION_END_DAY,
        Use_Weather_File_Holidays_and_Special_Days="Yes",
        Use_Weather_File_Daylight_Saving_Period="No",
        Apply_Weekend_Holiday_Rule="No",
        Use_Weather_File_Rain_Indicators="Yes",
        Use_Weather_File_Snow_Indicators="No",
    )
    print(f"[OK] RunPeriod set: {config.SIMULATION_START_MONTH}/{config.SIMULATION_START_DAY}"
          f" to {config.SIMULATION_END_MONTH}/{config.SIMULATION_END_DAY}")

    # -----------------------------------------------------------------------
    # 3. Declare Output:Variable entries for LLM sensor tools
    # -----------------------------------------------------------------------
    required_outputs = [
        "Zone Mean Air Temperature",
        "Zone People Occupant Count",
        "Zone Air System Sensible Cooling Rate",
        "Zone Air System Sensible Heating Rate",
        "Site Outdoor Air Drybulb Temperature",
    ]

    existing = {
        (o.Variable_Name, o.Key_Value)
        for o in idf.idfobjects["OUTPUT:VARIABLE"]
    }

    added = 0
    for var_name in required_outputs:
        if (var_name, "*") not in existing:
            idf.newidfobject(
                "OUTPUT:VARIABLE",
                Key_Value="*",
                Variable_Name=var_name,
                Reporting_Frequency="Timestep",
            )
            added += 1
    print(f"[OK] Added {added} Output:Variable declarations")

    # -----------------------------------------------------------------------
    # 4. Ensure 15-minute timestep
    # -----------------------------------------------------------------------
    for ts in idf.idfobjects["TIMESTEP"]:
        ts.Number_of_Timesteps_per_Hour = 4
    print(f"[OK] Timestep: 4 per hour (15 min)")

    # Save
    idf.save()
    print(f"[OK] Saved prepared IDF: {SOURCE_IDF.name}")


if __name__ == "__main__":
    prepare()