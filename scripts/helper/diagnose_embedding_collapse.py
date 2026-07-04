import sqlite3

conn = sqlite3.connect("models/state/index/ann_metadata_registry.db")

print("=== Overall distinct vessels indexed ===")
print(conn.execute("SELECT COUNT(DISTINCT vessel_id_int) FROM ann_node_registry").fetchone())

print("\n=== Per-superclass breakdown: row count vs distinct vessel count ===")
rows = conn.execute("""
    SELECT superclass_id,
           COUNT(*) AS n_rows,
           COUNT(DISTINCT vessel_id_int) AS n_distinct_vessels
    FROM ann_node_registry
    GROUP BY superclass_id
    ORDER BY n_rows DESC
""").fetchall()
for r in rows:
    print(r)

# Replace 0 with the actual superclass_id your evaluated vessel (96838952734631800) falls into
target_superclass = 0
print(f"\n=== Sample of vessel_ids in superclass {target_superclass} ===")
sample = conn.execute("""
    SELECT DISTINCT vessel_id_int FROM ann_node_registry
    WHERE superclass_id = ? LIMIT 20
""", (target_superclass,)).fetchall()
for s in sample:
    print(s)