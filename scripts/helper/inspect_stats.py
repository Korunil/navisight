# scripts/inspect_stats.py
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from navisight.pipeline.feature_registry import REGISTRY
except ImportError:
    print("❌ Error: Could not import REGISTRY. Ensure scripts/ is executed from the project root directory.")
    sys.exit(1)

def main():
    stats_path = "configs/global_stats.json"
    if not os.path.exists(stats_path):
        print(f"❌ Error: Configuration ledger file '{stats_path}' does not exist on disk.")
        return

    with open(stats_path, "r") as f:
        stats = json.load(f)

    # Extract verified dictionary targets from the lakehouse stats schema
    cohort_block = stats.get("global_cohort_scales", {}).get("0", {})
    weather_block = stats.get("weather_meso_scales", {})

    print("=" * 105)
    print("🛰️  NAVISIGHT DATA PLATFORM: COMPREHENSIVE SCHEMA ALIGNMENT LEDGER")
    print("=" * 105)
    print(f"{'Feature Name':<32} | {'Registry State':<16} | {'JSON Location Mapping':<30} | {'Numerical Payload':<20}")
    print("-" * 105)

    # Loop through every feature defined in the master code configuration
    for feat in REGISTRY.all_features:
        loc = "❌ MISSING FROM JSON"
        payload = "N/A"
        registry_state = "✅ Active"

        if feat in cohort_block:
            loc = "global_cohort_scales -> 0"
            p = cohort_block[feat]
            center = p.get('center', p.get('mean', 0.0))
            scale = p.get('scale', p.get('std', 1.0))
            payload = f"μ: {center:<6}, σ: {scale:<6}"
            
        elif feat in weather_block:
            loc = "weather_meso_scales"
            p = weather_block[feat]
            center = p.get('center', p.get('mean', 0.0))
            scale = p.get('scale', p.get('std', 1.0))
            payload = f"μ: {center:<6}, σ: {scale:<6}"
            
        elif feat in REGISTRY.maskable_for_loss:
            # If it's a maskable feature but completely omitted from scaling parameters
            registry_state = "⚠️ Unscaled Mask"

        print(f"{feat:<32} | {registry_state:<16} | {loc:<30} | {payload:<20}")

    print("=" * 105)

    # ── ADVANCED AUDIT: DETECT ORPHAN CHANNELS LEFT IN JSON ──
    print("\n🔍 AUDIT: CHECKING FOR ORPHAN FIELDS IN JSON (NOT DECLARED IN CODE REGISTRY)")
    print("-" * 105)
    orphans_found = False
    
    for feat in cohort_block:
        if feat not in REGISTRY.feature_index:
            print(f"🚨 ORPHAN DETECTED: Feature '{feat}' exists in global_cohort_scales['0'] but is missing from REGISTRY.all_features!")
            orphans_found = True
            
    for feat in weather_block:
        if feat not in REGISTRY.feature_index:
            print(f"🚨 ORPHAN DETECTED: Feature '{feat}' exists in weather_meso_scales but is missing from REGISTRY.all_features!")
            orphans_found = True
            
    if not orphans_found:
        print("✅ Clean Audit Pass: No orphan features found inside global_stats.json strings.")
    print("=" * 105)

if __name__ == "__main__":
    main()