"""CSV attendance report for one session: every enrolled student, PRESENT with time or ABSENT."""
import csv
import sys

import config
import database


def export(db, session_id):
    s = db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    rows = db.execute("""
        SELECT st.student_id, st.name, a.marked_at, COALESCE(a.status, 'ABSENT') AS status, a.similarity
        FROM students st
        LEFT JOIN attendance a ON a.student_id = st.student_id AND a.session_id = ?
        ORDER BY st.student_id""", (session_id,)).fetchall()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / f"session_{session_id}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "session_start", "session_end", "student_id", "name",
                    "status", "marked_at", "cosine_similarity"])
        for r in rows:
            sim = "" if r["similarity"] is None else f"{r['similarity']:.3f}"
            w.writerow([session_id, s["started_at"], s["ended_at"] or "", r["student_id"], r["name"],
                        r["status"], r["marked_at"] or "", sim])
    return path, rows


if __name__ == "__main__":
    db = database.connect()
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else db.execute("SELECT MAX(session_id) FROM sessions").fetchone()[0]
    if sid is None:
        raise SystemExit("no sessions yet")
    path, rows = export(db, sid)
    for r in rows:
        print(f"{r['student_id']:<8} {r['name']:<20} {r['status']:<8} {r['marked_at'] or ''}")
    print("saved", path)
