import sys
import os

EP_DIR = r"D:\EnergyPlus\Energy_Plus"
sys.path.insert(0, EP_DIR)

try:
    from pyenergyplus.api import EnergyPlusAPI
    print("[OK] pyenergyplus imported successfully")
except ImportError as e:
    print(f"[FAIL] Could not import pyenergyplus: {e}")
    sys.exit(1)

api = EnergyPlusAPI()
print(f"[OK] EnergyPlusAPI instantiated. Version: {api.api_version()}")

idf_path = os.path.join(EP_DIR, "ExampleFiles", "1ZoneUncontrolled.idf")

# Find any EPW file in WeatherData
wd = os.path.join(EP_DIR, "WeatherData")
epw_candidates = [f for f in os.listdir(wd) if f.endswith(".epw")]
if not epw_candidates:
    print(f"[FAIL] No .epw weather files found in {wd}")
    sys.exit(1)
epw_path = os.path.join(wd, epw_candidates[0])

if not os.path.exists(idf_path):
    print(f"[FAIL] IDF not found at {idf_path}")
    ef = os.path.join(EP_DIR, "ExampleFiles")
    print(f"First 5 IDFs in ExampleFiles: {[f for f in os.listdir(ef) if f.endswith('.idf')][:5]}")
    sys.exit(1)

print(f"[OK] Found IDF: {idf_path}")
print(f"[OK] Found EPW: {epw_path}")

output_dir = os.path.join(os.getcwd(), "smoke_output")
os.makedirs(output_dir, exist_ok=True)

print("[RUN] Starting EnergyPlus simulation...")
state = api.state_manager.new_state()
exit_code = api.runtime.run_energyplus(
    state,
    ['-w', epw_path, '-d', output_dir, idf_path]
)

if exit_code == 0:
    print(f"[SUCCESS] Simulation completed. Output in: {output_dir}")
    print(f"Files generated: {os.listdir(output_dir)}")
else:
    print(f"[FAIL] Simulation exited with code {exit_code}")
    sys.exit(1)