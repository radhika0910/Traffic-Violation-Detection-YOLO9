import sqlite3
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

class CloudDatabase:
    def __init__(self):
        self.connected = False
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 1. Setup Local SQLite (Backup/Audit)
        self.db_path = os.path.join(base_dir, "traffic_database.sqlite")
        self._init_sqlite()
        
        # 2. Setup Firebase Firestore
        cert_path = os.path.join(base_dir, "firebase_credentials.json")
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(cert_path)
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            self.connected = True
            print("[INFO] Connected to Firebase Firestore successfully.")
        except Exception as e:
            print(f"[WARNING] Firebase connection failed: {e}")
            print("[INFO] Falling back to Local SQL Database.")

    def _init_sqlite(self):
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
        status = "Recognized" if plate_text and plate_text != "UNKNOWN" and len(plate_text) >= 3 else "Unrecognized"
        timestamp_str = datetime.now().isoformat()
        
        # Log to SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO violations 
                (timestamp, license_plate, confidence, violation_type, status, snapshot_url, email_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp_str, plate_text, float(conf), violation_type, status, snapshot_url, False))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ERROR] SQLite write error: {e}")

        # Log to Firestore
        if self.connected:
            try:
                doc_ref = self.db.collection('violations').document()
                doc_ref.set({
                    'timestamp': timestamp_str,
                    'license_plate': plate_text,
                    'confidence': float(conf),
                    'violation_type': violation_type,
                    'status': status,
                    'snapshot_url': snapshot_url,
                    'email_sent': False,
                    'created_at': firestore.SERVER_TIMESTAMP
                })
                print(f"[CLOUD] Logged to Firestore: {plate_text}")
                return True
            except Exception as e:
                print(f"[ERROR] Firestore write error: {e}")
        return False

    def get_all_violations(self):
        # Fetch from Firestore if connected
        if self.connected:
            try:
                docs = self.db.collection('violations').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
                records = []
                for doc in docs:
                    data = doc.to_dict()
                    data['id'] = doc.id  # Use Firestore document ID
                    records.append(data)
                return records
            except Exception as e:
                print(f"[ERROR] Firestore read error: {e}")
        
        # Fallback to SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM violations ORDER BY timestamp DESC')
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
            conn.close()
            return records
        except Exception as e:
            print(f"[ERROR] SQL read error: {e}")
            return []
            
    def mark_email_sent(self, doc_id):
        # Update Firestore
        if self.connected:
            try:
                # Firestore doc_id is a string
                if isinstance(doc_id, str):
                    self.db.collection('violations').document(doc_id).update({'email_sent': True})
                    print(f"[CLOUD] Updated status for {doc_id}")
            except Exception as e:
                print(f"[ERROR] Firestore update error: {e}")

        # Update SQLite (always try to sync local)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # If doc_id is numeric, it's SQLite ID. If string, it's Firestore ID.
            # In some cases we might not be able to match them easily if we don't store Firestore ID in SQLite.
            # However, for the dashboard flow, doc_id passed from the UI will be what get_all_violations returned.
            if isinstance(doc_id, (int, float)) or (isinstance(doc_id, str) and doc_id.isdigit()):
                cursor.execute('UPDATE violations SET email_sent = ? WHERE id = ?', (True, doc_id))
                conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ERROR] SQLite update error: {e}")
