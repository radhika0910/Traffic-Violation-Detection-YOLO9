import sqlite3
import os
from datetime import datetime

class CloudDatabase:
    def __init__(self):
        self.connected = True
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(base_dir, "traffic_database.sqlite")
        self._init_db()
        print(f"[INFO] Connected to Local SQL Database at {self.db_path}.")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                license_plate TEXT,
                confidence REAL,
                violation_type TEXT,
                status TEXT,
                snapshot_url TEXT,
                email_sent BOOLEAN
            )
        ''')
        conn.commit()
        conn.close()

    def log_violation(self, plate_text, conf, violation_type, snapshot_url=""):
        try:
            status = "Recognized" if plate_text and plate_text != "UNKNOWN" and len(plate_text) >= 4 else "Unrecognized"
            timestamp_str = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO violations 
                (timestamp, license_plate, confidence, violation_type, status, snapshot_url, email_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp_str, plate_text, float(conf), violation_type, status, snapshot_url, False))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] SQL write error: {e}")
            return False

    def get_all_violations(self):
        try:
            conn = sqlite3.connect(self.db_path)
            # Row factory to return dicts instead of tuples
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM violations ORDER BY timestamp DESC')
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                data = dict(row)
                records.append(data)
                
            conn.close()
            return records
        except Exception as e:
            print(f"[ERROR] SQL read error: {e}")
            return []
            
    def mark_email_sent(self, doc_id):
         try:
             conn = sqlite3.connect(self.db_path)
             cursor = conn.cursor()
             cursor.execute('UPDATE violations SET email_sent = ? WHERE id = ?', (True, doc_id))
             conn.commit()
             conn.close()
         except Exception as e:
             print(f"[ERROR] SQL update error: {e}")
